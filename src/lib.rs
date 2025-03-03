//! This is the `rustpython` binary. If you're looking to embed RustPython into your application,
//! you're likely looking for the [`rustpython-vm`](https://docs.rs/rustpython-vm) crate.
//!
//! You can install `rustpython` with `cargo install rustpython`, or if you'd like to inject your
//! own native modules you can make a binary crate that depends on the `rustpython` crate (and
//! probably `rustpython-vm`, too), and make a `main.rs` that looks like:
//!
//! ```no_run
//! use rustpython_vm::{pymodule, py_freeze};
//! fn main() {
//!     rustpython::run(|vm| {
//!         vm.add_native_module("mymod".to_owned(), Box::new(mymod::make_module));
//!         vm.add_frozen(py_freeze!(source = "def foo(): pass", module_name = "otherthing"));
//!     });
//! }
//!
//! #[pymodule]
//! mod mymod {
//!     use rustpython_vm::builtins::PyStrRef;
//TODO: use rustpython_vm::prelude::*;
//!
//!     #[pyfunction]
//!     fn do_thing(x: i32) -> i32 {
//!         x + 1
//!     }
//!
//!     #[pyfunction]
//!     fn other_thing(s: PyStrRef) -> (String, usize) {
//!         let new_string = format!("hello from rust, {}!", s);
//!         let prev_len = s.as_str().len();
//!         (new_string, prev_len)
//!     }
//! }
//! ```
//!
//! The binary will have all the standard arguments of a python interpreter (including a REPL!) but
//! it will have your modules loaded into the vm.
#![allow(clippy::needless_doctest_main)]

#[macro_use]
extern crate clap;
extern crate env_logger;
#[macro_use]
extern crate log;

mod shell;

use clap::{App, AppSettings, Arg, ArgMatches};
use rustpython_vm::{scope::Scope, Interpreter, PyResult, Settings, VirtualMachine};
use std::{env, process::ExitCode, str::FromStr};

pub use rustpython_vm as vm;

/// The main cli of the `rustpython` interpreter. This function will return with `std::process::ExitCode`
/// based on the return code of the python code ran through the cli.
pub fn run<F>(init: F) -> ExitCode
where
    F: FnOnce(&mut VirtualMachine),
{
    #[cfg(feature = "flame-it")]
    let main_guard = flame::start_guard("RustPython main");
    env_logger::init();
    let app = App::new("RustPython");
    let matches = parse_arguments(app);
    let matches = &matches;
    let settings = create_settings(matches);

    // don't translate newlines (\r\n <=> \n)
    #[cfg(windows)]
    {
        extern "C" {
            fn _setmode(fd: i32, flags: i32) -> i32;
        }
        unsafe {
            _setmode(0, libc::O_BINARY);
            _setmode(1, libc::O_BINARY);
            _setmode(2, libc::O_BINARY);
        }
    }

    let interp = Interpreter::with_init(settings, |vm| {
        add_stdlib(vm);
        init(vm);
    });

    let exitcode = interp.run(move |vm| run_rustpython(vm, matches));

    #[cfg(feature = "flame-it")]
    {
        main_guard.end();
        if let Err(e) = write_profile(&matches) {
            error!("Error writing profile information: {}", e);
        }
    }
    ExitCode::from(exitcode)
}

fn parse_arguments<'a>(app: App<'a, '_>) -> ArgMatches<'a> {
    let app = app
        .setting(AppSettings::TrailingVarArg)
        .version(crate_version!())
        .author(crate_authors!())
        .about("Rust implementation of the Python language")
        .usage("rustpython [OPTIONS] [-c CMD | -m MODULE | FILE] [PYARGS]...")
        .arg(
            Arg::with_name("script")
                .required(false)
                .allow_hyphen_values(true)
                .multiple(true)
                .value_name("script, args")
                .min_values(1),
        )
        .arg(
            Arg::with_name("c")
                .short("c")
                .takes_value(true)
                .allow_hyphen_values(true)
                .multiple(true)
                .value_name("cmd, args")
                .min_values(1)
                .help("run the given string as a program"),
        )
        .arg(
            Arg::with_name("m")
                .short("m")
                .takes_value(true)
                .allow_hyphen_values(true)
                .multiple(true)
                .value_name("module, args")
                .min_values(1)
                .help("run library module as script"),
        )
        .arg(
            Arg::with_name("install_pip")
                .long("install-pip")
                .takes_value(true)
                .allow_hyphen_values(true)
                .multiple(true)
                .value_name("get-pip args")
                .min_values(0)
                .help("install the pip package manager for rustpython; \
                        requires rustpython be build with the ssl feature enabled."
                ),
        )
        .arg(
            Arg::with_name("optimize")
                .short("O")
                .multiple(true)
                .help("Optimize. Set __debug__ to false. Remove debug statements."),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .multiple(true)
                .help("Give the verbosity (can be applied multiple times)"),
        )
        .arg(Arg::with_name("debug").short("d").help("Debug the parser."))
        .arg(
            Arg::with_name("quiet")
                .short("q")
                .help("Be quiet at startup."),
        )
        .arg(
            Arg::with_name("inspect")
                .short("i")
                .help("Inspect interactively after running the script."),
        )
        .arg(
            Arg::with_name("no-user-site")
                .short("s")
                .help("don't add user site directory to sys.path."),
        )
        .arg(
            Arg::with_name("no-site")
                .short("S")
                .help("don't imply 'import site' on initialization"),
        )
        .arg(
            Arg::with_name("dont-write-bytecode")
                .short("B")
                .help("don't write .pyc files on import"),
        )
        .arg(
            Arg::with_name("ignore-environment")
                .short("E")
                .help("Ignore environment variables PYTHON* such as PYTHONPATH"),
        )
        .arg(
            Arg::with_name("isolate")
                .short("I")
                .help("isolate Python from the user's environment (implies -E and -s)"),
        )
        .arg(
            Arg::with_name("implementation-option")
                .short("X")
                .takes_value(true)
                .multiple(true)
                .number_of_values(1)
                .help("set implementation-specific option"),
        )
        .arg(
            Arg::with_name("warning-control")
                .short("W")
                .takes_value(true)
                .multiple(true)
                .number_of_values(1)
                .help("warning control; arg is action:message:category:module:lineno"),
        )
        .arg(
            Arg::with_name("check-hash-based-pycs")
                .long("check-hash-based-pycs")
                .takes_value(true)
                .number_of_values(1)
                .default_value("default")
                .help("always|default|never\ncontrol how Python invalidates hash-based .pyc files"),
        )
        .arg(
            Arg::with_name("bytes-warning")
                .short("b")
                .multiple(true)
                .help("issue warnings about using bytes where strings are usually expected (-bb: issue errors)"),
        ).arg(
            Arg::with_name("unbuffered")
                .short("u")
                .help(
                    "force the stdout and stderr streams to be unbuffered; \
                        this option has no effect on stdin; also PYTHONUNBUFFERED=x",
                ),
        );
    #[cfg(feature = "flame-it")]
    let app = app
        .arg(
            Arg::with_name("profile_output")
                .long("profile-output")
                .takes_value(true)
                .help("the file to output the profiling information to"),
        )
        .arg(
            Arg::with_name("profile_format")
                .long("profile-format")
                .takes_value(true)
                .help("the profile format to output the profiling information in"),
        );
    app.get_matches()
}

fn add_stdlib(vm: &mut VirtualMachine) {
    let _ = vm;
    #[cfg(feature = "stdlib")]
    vm.add_native_modules(rustpython_stdlib::get_module_inits());

    // if we're on freeze-stdlib, the core stdlib modules will be included anyway
    #[cfg(feature = "freeze-stdlib")]
    vm.add_frozen(rustpython_pylib::frozen_stdlib());

    #[cfg(not(feature = "freeze-stdlib"))]
    {
        use rustpython_vm::common::rc::PyRc;
        let state = PyRc::get_mut(&mut vm.state).unwrap();

        #[allow(clippy::needless_collect)] // false positive
        let path_list: Vec<_> = state.settings.path_list.drain(..).collect();

        // BUILDTIME_RUSTPYTHONPATH should be set when distributing
        if let Some(paths) = option_env!("BUILDTIME_RUSTPYTHONPATH") {
            state
                .settings
                .path_list
                .extend(split_paths(paths).map(|path| path.into_os_string().into_string().unwrap()))
        } else {
            #[cfg(feature = "rustpython-pylib")]
            state
                .settings
                .path_list
                .push(rustpython_pylib::LIB_PATH.to_owned())
        }

        state.settings.path_list.extend(path_list.into_iter());
    }
}

/// Create settings by examining command line arguments and environment
/// variables.
fn create_settings(matches: &ArgMatches) -> Settings {
    let mut settings = Settings::default();
    settings.isolated = matches.is_present("isolate");
    settings.ignore_environment = matches.is_present("ignore-environment");
    settings.interactive = !matches.is_present("c")
        && !matches.is_present("m")
        && (!matches.is_present("script") || matches.is_present("inspect"));
    settings.bytes_warning = matches.occurrences_of("bytes-warning");
    settings.no_site = matches.is_present("no-site");

    let ignore_environment = settings.ignore_environment || settings.isolated;

    if !ignore_environment {
        settings.path_list.extend(get_paths("RUSTPYTHONPATH"));
        settings.path_list.extend(get_paths("PYTHONPATH"));
    }

    // Now process command line flags:
    if matches.is_present("debug") || (!ignore_environment && env::var_os("PYTHONDEBUG").is_some())
    {
        settings.debug = true;
    }

    if matches.is_present("inspect")
        || (!ignore_environment && env::var_os("PYTHONINSPECT").is_some())
    {
        settings.inspect = true;
    }

    if matches.is_present("optimize") {
        settings.optimize = matches.occurrences_of("optimize").try_into().unwrap();
    } else if !ignore_environment {
        if let Ok(value) = get_env_var_value("PYTHONOPTIMIZE") {
            settings.optimize = value;
        }
    }

    if matches.is_present("verbose") {
        settings.verbose = matches.occurrences_of("verbose").try_into().unwrap();
    } else if !ignore_environment {
        if let Ok(value) = get_env_var_value("PYTHONVERBOSE") {
            settings.verbose = value;
        }
    }

    if matches.is_present("no-user-site")
        || matches.is_present("isolate")
        || (!ignore_environment && env::var_os("PYTHONNOUSERSITE").is_some())
    {
        settings.no_user_site = true;
    }

    if matches.is_present("quiet") {
        settings.quiet = true;
    }

    if matches.is_present("dont-write-bytecode")
        || (!ignore_environment && env::var_os("PYTHONDONTWRITEBYTECODE").is_some())
    {
        settings.dont_write_bytecode = true;
    }

    settings.check_hash_based_pycs = matches
        .value_of("check-hash-based-pycs")
        .unwrap_or("default")
        .to_owned();

    let mut dev_mode = false;
    let mut warn_default_encoding = false;
    if let Some(xopts) = matches.values_of("implementation-option") {
        settings.xopts.extend(xopts.map(|s| {
            let mut parts = s.splitn(2, '=');
            let name = parts.next().unwrap().to_owned();
            if name == "dev" {
                dev_mode = true
            }
            if name == "warn_default_encoding" {
                warn_default_encoding = true
            }
            let value = parts.next().map(ToOwned::to_owned);
            (name, value)
        }));
    }
    settings.dev_mode = dev_mode;
    if warn_default_encoding
        || (!ignore_environment && env::var_os("PYTHONWARNDEFAULTENCODING").is_some())
    {
        settings.warn_default_encoding = true;
    }

    if dev_mode {
        settings.warnopts.push("default".to_owned())
    }
    if settings.bytes_warning > 0 {
        let warn = if settings.bytes_warning > 1 {
            "error::BytesWarning"
        } else {
            "default::BytesWarning"
        };
        settings.warnopts.push(warn.to_owned());
    }
    if let Some(warnings) = matches.values_of("warning-control") {
        settings.warnopts.extend(warnings.map(ToOwned::to_owned));
    }

    let argv = if let Some(script) = matches.values_of("script") {
        script.map(ToOwned::to_owned).collect()
    } else if let Some(module) = matches.values_of("m") {
        std::iter::once("PLACEHOLDER".to_owned())
            .chain(module.skip(1).map(ToOwned::to_owned))
            .collect()
    } else if let Some(get_pip_args) = matches.values_of("install_pip") {
        settings.isolated = true;
        let mut args: Vec<_> = get_pip_args.map(ToOwned::to_owned).collect();
        if args.is_empty() {
            args.push("ensurepip".to_owned());
            args.push("--upgrade".to_owned());
            args.push("--default-pip".to_owned());
        }
        match args.first().map(String::as_str) {
            Some("ensurepip") | Some("get-pip") => (),
            _ => panic!("--install-pip takes ensurepip or get-pip as first argument"),
        }
        args
    } else if let Some(cmd) = matches.values_of("c") {
        std::iter::once("-c".to_owned())
            .chain(cmd.skip(1).map(ToOwned::to_owned))
            .collect()
    } else {
        vec!["".to_owned()]
    };

    let hash_seed = match env::var("PYTHONHASHSEED") {
        Ok(s) if s == "random" => Some(None),
        Ok(s) => s.parse::<u32>().ok().map(Some),
        Err(_) => Some(None),
    };
    settings.hash_seed = hash_seed.unwrap_or_else(|| {
        error!("Fatal Python init error: PYTHONHASHSEED must be \"random\" or an integer in range [0; 4294967295]");
        // TODO: Need to change to ExitCode or Termination
        std::process::exit(1)
    });

    settings.argv = argv;

    settings
}

/// Get environment variable and turn it into integer.
fn get_env_var_value(name: &str) -> Result<u8, std::env::VarError> {
    env::var(name).map(|value| {
        if let Ok(value) = u8::from_str(&value) {
            value
        } else {
            1
        }
    })
}

/// Helper function to retrieve a sequence of paths from an environment variable.
fn get_paths(env_variable_name: &str) -> impl Iterator<Item = String> + '_ {
    env::var_os(env_variable_name)
        .into_iter()
        .flat_map(move |paths| {
            split_paths(&paths)
                .map(|path| {
                    path.into_os_string()
                        .into_string()
                        .unwrap_or_else(|_| panic!("{} isn't valid unicode", env_variable_name))
                })
                .collect::<Vec<_>>()
        })
}
#[cfg(not(target_os = "wasi"))]
use env::split_paths;
#[cfg(target_os = "wasi")]
fn split_paths<T: AsRef<std::ffi::OsStr> + ?Sized>(
    s: &T,
) -> impl Iterator<Item = std::path::PathBuf> + '_ {
    use std::os::wasi::ffi::OsStrExt;
    let s = s.as_ref().as_bytes();
    s.split(|b| *b == b':')
        .map(|x| std::ffi::OsStr::from_bytes(x).to_owned().into())
}

#[cfg(feature = "flame-it")]
fn write_profile(matches: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs, io};

    enum ProfileFormat {
        Html,
        Text,
        Speedscope,
    }

    let profile_output = matches.value_of_os("profile_output");

    let profile_format = match matches.value_of("profile_format") {
        Some("html") => ProfileFormat::Html,
        Some("text") => ProfileFormat::Text,
        None if profile_output == Some("-".as_ref()) => ProfileFormat::Text,
        Some("speedscope") | None => ProfileFormat::Speedscope,
        Some(other) => {
            error!("Unknown profile format {}", other);
            // TODO: Need to change to ExitCode or Termination
            std::process::exit(1);
        }
    };

    let profile_output = profile_output.unwrap_or_else(|| match profile_format {
        ProfileFormat::Html => "flame-graph.html".as_ref(),
        ProfileFormat::Text => "flame.txt".as_ref(),
        ProfileFormat::Speedscope => "flamescope.json".as_ref(),
    });

    let profile_output: Box<dyn io::Write> = if profile_output == "-" {
        Box::new(io::stdout())
    } else {
        Box::new(fs::File::create(profile_output)?)
    };

    let profile_output = io::BufWriter::new(profile_output);

    match profile_format {
        ProfileFormat::Html => flame::dump_html(profile_output)?,
        ProfileFormat::Text => flame::dump_text_to_writer(profile_output)?,
        ProfileFormat::Speedscope => flamescope::dump(profile_output)?,
    }

    Ok(())
}

fn setup_main_module(vm: &VirtualMachine) -> PyResult<Scope> {
    let scope = vm.new_scope_with_builtins();
    let main_module = vm.new_module("__main__", scope.globals.clone(), None);
    main_module
        .dict()
        .and_then(|d| {
            d.set_item("__annotations__", vm.ctx.new_dict().into(), vm)
                .ok()
        })
        .expect("Failed to initialize __main__.__annotations__");

    vm.sys_module
        .clone()
        .get_attr("modules", vm)?
        .set_item("__main__", main_module, vm)?;

    Ok(scope)
}

#[cfg(feature = "ssl")]
fn get_pip(scope: Scope, vm: &VirtualMachine) -> PyResult<()> {
    let get_getpip = rustpython_vm::py_compile!(
        source = r#"\
__import__("io").TextIOWrapper(
    __import__("urllib.request").request.urlopen("https://bootstrap.pypa.io/get-pip.py")
).read()
"#,
        mode = "eval"
    );
    eprintln!("downloading get-pip.py...");
    let getpip_code = vm.run_code_obj(vm.ctx.new_code(get_getpip), scope.clone())?;
    let getpip_code: rustpython_vm::builtins::PyStrRef = getpip_code
        .downcast()
        .expect("TextIOWrapper.read() should return str");
    eprintln!("running get-pip.py...");
    vm.run_code_string(scope, getpip_code.as_str(), "get-pip.py".to_owned())?;
    Ok(())
}

#[cfg(feature = "ssl")]
fn ensurepip(_: Scope, vm: &VirtualMachine) -> PyResult<()> {
    vm.run_module("ensurepip")
}

fn install_pip(_scope: Scope, vm: &VirtualMachine) -> PyResult<()> {
    #[cfg(feature = "ssl")]
    {
        match vm.state.settings.argv[0].as_str() {
            "ensurepip" => ensurepip(_scope, vm),
            "get-pip" => get_pip(_scope, vm),
            _ => unreachable!(),
        }
    }

    #[cfg(not(feature = "ssl"))]
    Err(vm.new_exception_msg(
        vm.ctx.exceptions.system_error.to_owned(),
        "install-pip requires rustpython be build with '--features=ssl'".to_owned(),
    ))
}

fn run_rustpython(vm: &VirtualMachine, matches: &ArgMatches) -> PyResult<()> {
    let scope = setup_main_module(vm)?;

    let site_result = vm.import("site", None, 0);

    if site_result.is_err() {
        warn!(
            "Failed to import site, consider adding the Lib directory to your RUSTPYTHONPATH \
             environment variable",
        );
    }

    // Figure out if a -c option was given:
    if let Some(command) = matches.value_of("c") {
        debug!("Running command {}", command);
        vm.run_code_string(scope, command, "<stdin>".to_owned())?;
    } else if let Some(module) = matches.value_of("m") {
        debug!("Running module {}", module);
        vm.run_module(module)?;
    } else if matches.is_present("install_pip") {
        install_pip(scope, vm)?;
    } else if let Some(filename) = matches.value_of("script") {
        debug!("Running file {}", filename);
        vm.run_script(scope.clone(), filename)?;
        if matches.is_present("inspect") {
            shell::run_shell(vm, scope)?;
        }
    } else {
        println!(
            "Welcome to the magnificent Rust Python {} interpreter \u{1f631} \u{1f596}",
            crate_version!()
        );
        shell::run_shell(vm, scope)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interpreter() -> Interpreter {
        Interpreter::with_init(Settings::default(), |vm| {
            add_stdlib(vm);
        })
    }

    #[test]
    fn test_run_script() {
        interpreter().enter(|vm| {
            vm.unwrap_pyresult((|| {
                let scope = setup_main_module(vm)?;
                // test file run
                vm.run_script(scope, "extra_tests/snippets/dir_main/__main__.py")?;

                let scope = setup_main_module(vm)?;
                // test module run
                vm.run_script(scope, "extra_tests/snippets/dir_main")?;

                Ok(())
            })());
        })
    }
}
