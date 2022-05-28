//! mmap module
pub(crate) use mmap::make_module;

#[pymodule]
mod mmap {
    use crate::vm::{
        builtins::PyBytes,
        builtins::PyIntRef,
        builtins::PyTypeRef,
        function::OptionalArg,
        function::OptionalOption,
        types::Constructor,
        types::{AsBuffer, AsMapping, AsSequence},
        ByteInnerFindOptions, FromArgs, PyObject, PyPayload, PyRef, PyResult,
        TryFromBorrowedObject, VirtualMachine,
    };
    use crossbeam_utils::atomic::AtomicCell;
    use memmap2::{Advice, Mmap, MmapMut, MmapOptions};
    use num_traits::{Signed, ToPrimitive};
    use std::fs::File;
    use std::os::unix::io::{FromRawFd, IntoRawFd, RawFd};

    #[repr(C)]
    #[derive(PartialEq, Eq, Debug)]
    enum AccessMode {
        Default = 0,
        Read = 1,
        Write = 2,
        Copy = 3,
    }

    impl TryFromBorrowedObject for AccessMode {
        fn try_from_borrowed_object(vm: &VirtualMachine, obj: &PyObject) -> PyResult<Self> {
            let i = u32::try_from_borrowed_object(vm, obj)?;
            Ok(match i {
                0 => Self::Default,
                1 => Self::Read,
                2 => Self::Write,
                3 => Self::Copy,
                _ => return Err(vm.new_value_error("Not a valid AccessMode value".to_owned())),
            })
        }
    }

    #[pyattr]
    use libc::{
        MADV_DONTNEED, MADV_FREE_REUSABLE, MADV_FREE_REUSE, MADV_NORMAL, MADV_RANDOM,
        MADV_SEQUENTIAL, MADV_WILLNEED, MAP_ANON, MAP_ANONYMOUS, MAP_PRIVATE, MAP_SHARED,
        PROT_READ, PROT_WRITE,
    };

    #[cfg(target_os = "linux")]
    use libc::{
        MADV_DODUMP, MADV_DOFORK, MADV_DONTDUMP, MADV_DONTFORK, MADV_FREE, MADV_HUGEPAGE,
        MADV_HWPOISON, MADV_MERGEABLE, MADV_NOHUGEPAGE, MADV_REMOVE, MADV_SOFT_OFFLINE,
        MADV_UNMERGEABLE,
    };

    #[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
    use libc::{MAP_DENYWRITE, MAP_EXECUTABLE, MAP_POPULATE};

    #[pyattr]
    const ACCESS_DEFAULT: u32 = AccessMode::Default as u32;
    #[pyattr]
    const ACCESS_READ: u32 = AccessMode::Read as u32;
    #[pyattr]
    const ACCESS_WRITE: u32 = AccessMode::Write as u32;
    #[pyattr]
    const ACCESS_COPY: u32 = AccessMode::Copy as u32;

    #[pyattr(name = "PAGESIZE")]
    fn pagesize(vm: &VirtualMachine) -> usize {
        page_size::get()
    }

    #[derive(Debug)]
    enum MmapObj {
        Write(MmapMut),
        Read(Mmap),
    }

    #[pyattr]
    #[pyclass(name = "mmap")]
    #[derive(Debug, PyPayload)]
    struct PyMmap {
        closed: AtomicCell<bool>,
        mmap: MmapObj,
        fd: RawFd,
        offset: isize,
        size: isize,
        pos: isize, // relative to offset
        exports: usize,
        //     PyObject *weakreflist;
        access: AccessMode,
    }

    #[derive(FromArgs)]
    struct MmapNewArgs {
        #[pyarg(any)]
        fileno: RawFd,
        #[pyarg(any)]
        length: isize,
        #[pyarg(any, default = "MAP_SHARED")]
        flags: libc::c_int,
        #[pyarg(any, default = "PROT_WRITE|PROT_READ")]
        prot: libc::c_int,
        #[pyarg(any, default = "AccessMode::Default")]
        access: AccessMode,
        #[pyarg(any, default = "0")]
        offset: isize,
    }

    fn saturate_to_isize(py_int: PyIntRef) -> isize {
        let big = py_int.as_bigint();
        big.to_isize().unwrap_or_else(|| {
            if big.is_negative() {
                isize::MIN
            } else {
                isize::MAX
            }
        })
    }

    #[derive(FromArgs)]
    pub struct FlushOptions {
        #[pyarg(positional, default)]
        offset: Option<PyIntRef>,
        #[pyarg(positional, default)]
        size: Option<PyIntRef>,
    }

    #[derive(FromArgs)]
    pub struct FindOptions {
        #[pyarg(positional)]
        sub: Vec<u8>,
        #[pyarg(positional, default)]
        start: Option<PyIntRef>,
        #[pyarg(positional, default)]
        end: Option<PyIntRef>,
    }

    #[derive(FromArgs)]
    pub struct AdviseOptions {
        #[pyarg(positional)]
        option: libc::c_int,
        #[pyarg(positional, default)]
        start: Option<PyIntRef>,
        #[pyarg(positional, default)]
        length: Option<PyIntRef>,
    }

    impl Constructor for PyMmap {
        type Args = MmapNewArgs;

        // TODO: Windows is not supported right now.
        fn py_new(
            cls: PyTypeRef,
            MmapNewArgs {
                fileno: mut fd,
                length,
                flags,
                prot,
                access,
                offset,
            }: Self::Args,
            vm: &VirtualMachine,
        ) -> PyResult {
            let mut map_size = length;
            if map_size < 0 {
                return Err(
                    vm.new_overflow_error("memory mapped length must be positive".to_owned())
                );
            }

            if offset < 0 {
                return Err(
                    vm.new_overflow_error("memory mapped offset must be positive".to_owned())
                );
            }

            if (access != AccessMode::Default)
                && ((flags != MAP_SHARED) || (prot != (PROT_WRITE | PROT_READ)))
            {
                return Err(vm.new_value_error(
                    "mmap can't specify both access and flags, prot.".to_owned(),
                ));
            }

            let (flags, prot, access) = match access {
                AccessMode::Read => (MAP_SHARED, PROT_READ, access),
                AccessMode::Write => (MAP_SHARED, PROT_READ | PROT_WRITE, access),
                AccessMode::Copy => (MAP_PRIVATE, PROT_READ | PROT_WRITE, access),
                AccessMode::Default => {
                    let access = if (prot & PROT_READ) != 0 && (prot & PROT_WRITE) != 0 {
                        access
                    } else if (prot & PROT_WRITE) != 0 {
                        AccessMode::Write
                    } else {
                        AccessMode::Read
                    };
                    (flags, prot, access)
                }
                _ => return Err(vm.new_value_error("mmap invalid access parameter.".to_owned())),
            };

            if fd != -1 {
                let file = unsafe { File::from_raw_fd(fd) };
                let file_len = match file.metadata() {
                    Ok(m) => m.len() as isize,
                    Err(e) => return Err(vm.new_os_error(e.to_string())),
                };
                // File::from_raw_fd will consume the fd, so we
                // have to  get it again.
                fd = file.into_raw_fd();
                if map_size == 0 {
                    if file_len == 0 {
                        return Err(vm.new_value_error("cannot mmap an empty file".to_owned()));
                    }

                    if offset > file_len {
                        return Err(
                            vm.new_value_error("mmap offset is greater than file size".to_owned())
                        );
                    }

                    map_size = file_len - offset;
                } else if offset > file_len || file_len - offset < map_size {
                    return Err(
                        vm.new_value_error("mmap length is greater than file size".to_owned())
                    );
                }
            }

            let mut mmap_opt = MmapOptions::new();
            let mmap_opt = mmap_opt.offset(offset.try_into().unwrap());

            let mmap = match access {
                AccessMode::Default | AccessMode::Write => MmapObj::Write(
                    unsafe { mmap_opt.map_mut(fd) }.map_err(|e| vm.new_os_error(e.to_string()))?,
                ),
                AccessMode::Read => MmapObj::Read(
                    unsafe { mmap_opt.map(fd) }.map_err(|e| vm.new_os_error(e.to_string()))?,
                ),
                AccessMode::Copy => MmapObj::Write(
                    unsafe { mmap_opt.map_copy(fd) }.map_err(|e| vm.new_os_error(e.to_string()))?,
                ),
            };

            let m_obj = Self {
                closed: AtomicCell::new(false),
                mmap,
                fd,
                offset,
                size: map_size.into(),
                pos: 0,
                exports: 0,
                access,
            };

            m_obj.into_ref_with_type(vm, cls).map(Into::into)
        }
    }

    /*
        impl AsBuffer for PyMmap {
        fn as_buffer(zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<PyBuffer> {
        let buf = PyBuffer::new(
        zelf.to_owned().into(),
        BufferDescriptor::simple(zelf.len(), true),
        &BUFFER_METHODS,
    );
        Ok(buf)
    }
    }

        impl AsMapping for PyMmap {
        fn as_mapping(_zelf: &Py<Self>, _vm: &VirtualMachine) -> PyMappingMethods {
        Self::MAPPING_METHODS
    }
    }

        impl AsSequence for PyMmap {
        fn as_sequence(_zelf: &Py<Self>, _vm: &VirtualMachine) -> Cow<'static, PySequenceMethods> {
        Cow::Borrowed(&Self::SEQUENCE_METHODS)
    }
    }
         */

    #[pyimpl(with(
        Constructor,
        //AsMapping,
        //AsSequence,
        //AsBuffer
    ))]
    impl PyMmap {
        #[pyproperty]
        fn closed(&self) -> bool {
            self.closed.load()
        }

        #[pymethod]
        fn close(&self, vm: &VirtualMachine) -> PyResult<()> {
            if self.closed() {
                return Ok(());
            }

            if self.exports > 0 {
                return Err(vm.new_value_error("cannot close exported pointers exist.".to_owned()));
            }
            self.closed.store(true);

            //TODO drop mmap object
            Ok(())
        }

        #[pymethod]
        fn find(&self, options: FindOptions, vm: &VirtualMachine) -> PyResult<Option<usize>> {
            let sub = &options.sub;
            match &self.mmap {
                MmapObj::Read(mmap) => Ok(mmap.windows(sub.len()).position(|window| window == sub)),
                MmapObj::Write(mmap) => {
                    Ok(mmap.windows(sub.len()).position(|window| window == sub))
                }
            }
        }

        #[pymethod]
        fn rfind(&self, options: FindOptions, vm: &VirtualMachine) -> PyResult<Option<usize>> {
            let sub = &options.sub;
            match &self.mmap {
                MmapObj::Read(mmap) => {
                    Ok(mmap.windows(sub.len()).rposition(|window| window == sub))
                }
                MmapObj::Write(mmap) => {
                    Ok(mmap.windows(sub.len()).rposition(|window| window == sub))
                }
            }
        }

        #[pymethod]
        fn flush(&self, options: FlushOptions, vm: &VirtualMachine) -> PyResult<()> {
            let offset = options.offset.map_or(0, saturate_to_isize);
            let size = options.size.map_or(self.size, saturate_to_isize);

            if size < 0 || offset < 0 || self.size - offset < size {
                return Err(vm.new_value_error("flush values out of range".to_owned()));
            }

            if self.access == AccessMode::Read || self.access == AccessMode::Copy {
                return Ok(());
            }

            match &self.mmap {
                MmapObj::Read(mmap) => {}
                MmapObj::Write(mmap) => {
                    if let Err(e) = mmap.flush_range(offset as usize, size as usize) {
                        return Err(vm.new_os_error(e.to_string()));
                    }
                }
            }

            Ok(())
        }

        #[pymethod]
        fn madvise(&self, options: AdviseOptions) -> PyResult<()> {
            Ok(())
        }

        #[pymethod]
        fn r#move(&self, dest: isize, src: isize, count: usize) -> PyResult<()> {
            Ok(())
        }

        #[pymethod]
        fn read(&self, n: OptionalOption<usize>) -> PyResult<()> {
            Ok(())
        }

        #[pymethod]
        fn read_byte(&self) -> PyResult<()> {
            Ok(())
        }

        #[pymethod]
        fn readline(&self) -> PyResult<()> {
            Ok(())
        }

        #[pymethod]
        fn resize(&self, newsize: usize) -> PyResult<()> {
            Ok(())
        }

        #[pymethod]
        fn seek(&self, pos: isize, whence: OptionalOption<libc::c_int>) -> PyResult<()> {
            Ok(())
        }

        #[pymethod]
        fn size(&self) -> PyResult<isize> {
            Ok(self.size)
        }

        #[pymethod]
        fn tell(&self) -> PyResult<isize> {
            Ok(self.pos)
        }

        #[pymethod]
        fn write(&self, bytes: PyBytes) -> PyResult<isize> {
            Ok(self.pos)
        }

        #[pymethod]
        fn write_byte(&self, byte: u8) -> PyResult<isize> {
            Ok(self.pos)
        }

        //     {"find",            (PyCFunction) mmap_find_method,         METH_VARARGS},
        //     {"rfind",           (PyCFunction) mmap_rfind_method,        METH_VARARGS},
        //     {"flush",           (PyCFunction) mmap_flush_method,        METH_VARARGS},
        // #ifdef HAVE_MADVISE
        //     {"madvise",         (PyCFunction) mmap_madvise_method,      METH_VARARGS},
        // #endif
        //     {"move",            (PyCFunction) mmap_move_method,         METH_VARARGS},
        //     {"read",            (PyCFunction) mmap_read_method,         METH_VARARGS},
        //     {"read_byte",       (PyCFunction) mmap_read_byte_method,    METH_NOARGS},
        //     {"readline",        (PyCFunction) mmap_read_line_method,    METH_NOARGS},
        //     {"resize",          (PyCFunction) mmap_resize_method,       METH_VARARGS},
        //     {"seek",            (PyCFunction) mmap_seek_method,         METH_VARARGS},
        //     {"size",            (PyCFunction) mmap_size_method,         METH_NOARGS},
        //     {"tell",            (PyCFunction) mmap_tell_method,         METH_NOARGS},
        //     {"write",           (PyCFunction) mmap_write_method,        METH_VARARGS},
        //     {"write_byte",      (PyCFunction) mmap_write_byte_method,   METH_VARARGS},
        //     {"__enter__",       (PyCFunction) mmap__enter__method,      METH_NOARGS},
        //     {"__exit__",        (PyCFunction) mmap__exit__method,       METH_VARARGS},
        // #ifdef MS_WINDOWS
        //     {"__sizeof__",      (PyCFunction) mmap__sizeof__method,     METH_NOARGS},
        // #endif
    }
}
