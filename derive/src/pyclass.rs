use super::Diagnostic;
use crate::util::{
    path_eq, pyclass_ident_and_attrs, text_signature, ClassItemMeta, ContentItem, ContentItemInner,
    ErrorVec, ItemMeta, ItemMetaInner, ItemNursery, SimpleItemMeta, ALL_ALLOWED_NAMES,
};
use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned, ToTokens};
use std::collections::HashMap;
use std::str::FromStr;
use syn::{
    parse::{Parse, ParseStream, Result as ParsingResult},
    parse_quote,
    spanned::Spanned,
    Attribute, AttributeArgs, Ident, Item, LitStr, Meta, NestedMeta, Result, Token,
};
use syn_ext::ext::*;

#[derive(Copy, Clone, Debug)]
enum AttrName {
    Method,
    ClassMethod,
    StaticMethod,
    GetSet,
    Slot,
    Attr,
    ExtendClass,
    Member,
}

impl std::fmt::Display for AttrName {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match self {
            Self::Method => "pymethod",
            Self::ClassMethod => "pyclassmethod",
            Self::StaticMethod => "pystaticmethod",
            Self::GetSet => "pyproperty",
            Self::Slot => "pyslot",
            Self::Attr => "pyattr",
            Self::ExtendClass => "extend_class",
            Self::Member => "pymember",
        };
        s.fmt(f)
    }
}

impl FromStr for AttrName {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s {
            "pymethod" => Self::Method,
            "pyclassmethod" => Self::ClassMethod,
            "pystaticmethod" => Self::StaticMethod,
            "pyproperty" => Self::GetSet,
            "pyslot" => Self::Slot,
            "pyattr" => Self::Attr,
            "extend_class" => Self::ExtendClass,
            "pymember" => Self::Member,
            s => {
                return Err(s.to_owned());
            }
        })
    }
}

#[derive(Default)]
struct ImplContext {
    impl_extend_items: ItemNursery,
    getset_items: GetSetNursery,
    extend_slots_items: ItemNursery,
    class_extensions: Vec<TokenStream>,
    errors: Vec<syn::Error>,
}

fn extract_items_into_context<'a, Item>(
    context: &mut ImplContext,
    items: impl Iterator<Item = &'a mut Item>,
) where
    Item: ItemLike + ToTokens + GetIdent + syn_ext::ext::ItemAttrExt + 'a,
{
    for item in items {
        let r = item.try_split_attr_mut(|attrs, item| {
            let (pyitems, cfgs) = attrs_to_content_items(attrs, impl_item_new::<Item>)?;
            for pyitem in pyitems.iter().rev() {
                let r = pyitem.gen_impl_item(ImplItemArgs::<Item> {
                    item,
                    attrs,
                    context,
                    cfgs: cfgs.as_slice(),
                });
                context.errors.ok_or_push(r);
            }
            Ok(())
        });
        context.errors.ok_or_push(r);
    }
    context.errors.ok_or_push(context.getset_items.validate());
}

pub(crate) fn impl_pyimpl(attr: AttributeArgs, item: Item) -> Result<TokenStream> {
    let mut context = ImplContext::default();
    let mut tokens = match item {
        Item::Impl(mut imp) => {
            extract_items_into_context(&mut context, imp.items.iter_mut());

            let ty = &imp.self_ty;
            let ExtractedImplAttrs {
                with_impl,
                flags,
                with_slots,
            } = extract_impl_attrs(attr, &Ident::new(&quote!(ty).to_string(), ty.span()))?;

            let getset_impl = &context.getset_items;
            let extend_impl = context.impl_extend_items.validate()?;
            let slots_impl = context.extend_slots_items.validate()?;
            let class_extensions = &context.class_extensions;
            quote! {
                #imp
                impl ::rustpython_vm::class::PyClassImpl for #ty {
                    const TP_FLAGS: ::rustpython_vm::types::PyTypeFlags = #flags;

                    fn impl_extend_class(
                        ctx: &::rustpython_vm::Context,
                        class: &'static ::rustpython_vm::Py<::rustpython_vm::builtins::PyType>,
                    ) {
                        #getset_impl
                        #extend_impl
                        #with_impl
                        #(#class_extensions)*
                    }

                    fn extend_slots(slots: &mut ::rustpython_vm::types::PyTypeSlots) {
                        #with_slots
                        #slots_impl
                    }
                }
            }
        }
        Item::Trait(mut trai) => {
            let mut context = ImplContext::default();
            extract_items_into_context(&mut context, trai.items.iter_mut());

            let ExtractedImplAttrs {
                with_impl,
                with_slots,
                ..
            } = extract_impl_attrs(attr, &trai.ident)?;

            let getset_impl = &context.getset_items;
            let extend_impl = &context.impl_extend_items.validate()?;
            let slots_impl = &context.extend_slots_items.validate()?;
            let class_extensions = &context.class_extensions;
            let extra_methods = iter_chain![
                parse_quote! {
                    fn __extend_py_class(
                        ctx: &::rustpython_vm::Context,
                        class: &'static ::rustpython_vm::Py<::rustpython_vm::builtins::PyType>,
                    ) {
                        #getset_impl
                        #extend_impl
                        #with_impl
                        #(#class_extensions)*
                    }
                },
                parse_quote! {
                    fn __extend_slots(slots: &mut ::rustpython_vm::types::PyTypeSlots) {
                        #with_slots
                        #slots_impl
                    }
                },
            ];
            trai.items.extend(extra_methods);

            trai.into_token_stream()
        }
        item => item.into_token_stream(),
    };
    if let Some(error) = context.errors.into_error() {
        let error = Diagnostic::from(error);
        tokens = quote! {
            #tokens
            #error
        }
    }
    Ok(tokens)
}

fn generate_class_def(
    ident: &Ident,
    name: &str,
    module_name: Option<&str>,
    base: Option<String>,
    metaclass: Option<String>,
    attrs: &[Attribute],
) -> Result<TokenStream> {
    let doc = attrs.doc().or_else(|| {
        let module_name = module_name.unwrap_or("builtins");
        crate::doc::Database::shared()
            .try_module_item(module_name, name)
            .ok()
            .flatten()
            .map(str::to_owned)
    });
    let doc = if let Some(doc) = doc {
        quote!(Some(#doc))
    } else {
        quote!(None)
    };
    let module_class_name = if let Some(module_name) = module_name {
        format!("{}.{}", module_name, name)
    } else {
        name.to_owned()
    };
    let module_name = match module_name {
        Some(v) => quote!(Some(#v) ),
        None => quote!(None),
    };
    let basicsize = quote!(std::mem::size_of::<#ident>());
    let is_pystruct = attrs.iter().any(|attr| {
        path_eq(&attr.path, "derive")
            && if let Ok(Meta::List(l)) = attr.parse_meta() {
                l.nested
                    .into_iter()
                    .any(|n| n.get_ident().map_or(false, |p| p == "PyStructSequence"))
            } else {
                false
            }
    });
    if base.is_some() && is_pystruct {
        return Err(syn::Error::new_spanned(
            ident,
            "PyStructSequence cannot have `base` class attr",
        ));
    }
    let base_class = if is_pystruct {
        Some(quote! { rustpython_vm::builtins::PyTuple })
    } else {
        base.map(|typ| {
            let typ = Ident::new(&typ, ident.span());
            quote_spanned! { ident.span() => #typ }
        })
    }
    .map(|typ| {
        quote! {
            fn static_baseclass() -> &'static ::rustpython_vm::Py<::rustpython_vm::builtins::PyType> {
                use rustpython_vm::class::StaticType;
                #typ::static_type()
            }
        }
    });

    let meta_class = metaclass.map(|typ| {
        let typ = Ident::new(&typ, ident.span());
        quote! {
            fn static_metaclass() -> &'static ::rustpython_vm::Py<::rustpython_vm::builtins::PyType> {
                use rustpython_vm::class::StaticType;
                #typ::static_type()
            }
        }
    });

    let tokens = quote! {
        impl ::rustpython_vm::class::PyClassDef for #ident {
            const NAME: &'static str = #name;
            const MODULE_NAME: Option<&'static str> = #module_name;
            const TP_NAME: &'static str = #module_class_name;
            const DOC: Option<&'static str> = #doc;
            const BASICSIZE: usize = #basicsize;
        }

        impl ::rustpython_vm::class::StaticType for #ident {
            fn static_cell() -> &'static ::rustpython_vm::common::static_cell::StaticCell<::rustpython_vm::builtins::PyTypeRef> {
                ::rustpython_vm::common::static_cell! {
                    static CELL: ::rustpython_vm::builtins::PyTypeRef;
                }
                &CELL
            }

            #meta_class

            #base_class
        }
    };
    Ok(tokens)
}

pub(crate) fn impl_pyclass(attr: AttributeArgs, item: Item) -> Result<TokenStream> {
    if matches!(item, syn::Item::Use(_)) {
        return Ok(quote!(#item));
    }
    let (ident, attrs) = pyclass_ident_and_attrs(&item)?;
    let fake_ident = Ident::new("pyclass", item.span());
    let class_meta = ClassItemMeta::from_nested(ident.clone(), fake_ident, attr.into_iter())?;
    let class_name = class_meta.class_name()?;
    let module_name = class_meta.module()?;
    let base = class_meta.base()?;
    let metaclass = class_meta.metaclass()?;
    let class_def = generate_class_def(
        ident,
        &class_name,
        module_name.as_deref(),
        base,
        metaclass,
        attrs,
    )?;

    let ret = quote! {
        #item
        #class_def
    };
    Ok(ret)
}

/// Special macro to create exception types.
///
/// Why do we need it and why can't we just use `pyclass` macro instead?
/// We generate exception types with a `macro_rules`,
/// similar to how CPython does it.
/// But, inside `macro_rules` we don't have an opportunity
/// to add non-literal attributes to `pyclass`.
/// That's why we have to use this proxy.
pub(crate) fn impl_pyexception(attr: AttributeArgs, item: Item) -> Result<TokenStream> {
    let class_name = parse_vec_ident(&attr, &item, 0, "first 'class_name'")?;
    let base_class_name = parse_vec_ident(&attr, &item, 1, "second 'base_class_name'")?;

    // We also need to strip `Py` prefix from `class_name`,
    // due to implementation and Python naming conventions mismatch:
    // `PyKeyboardInterrupt` -> `KeyboardInterrupt`
    let class_name = class_name.strip_prefix("Py").ok_or_else(|| {
        syn::Error::new_spanned(&item, "We require 'class_name' to have 'Py' prefix")
    })?;

    // We just "proxy" it into `pyclass` macro, because, exception is a class.
    let ret = quote! {
        #[pyclass(module = false, name = #class_name, base = #base_class_name)]
        #item
    };
    Ok(ret)
}

pub(crate) fn impl_define_exception(exc_def: PyExceptionDef) -> Result<TokenStream> {
    let PyExceptionDef {
        class_name,
        base_class,
        ctx_name,
        docs,
        slot_new,
        init,
    } = exc_def;

    // We need this method, because of how `CPython` copies `__new__`
    // from `BaseException` in `SimpleExtendsException` macro.
    // See: `BaseException_new`
    let slot_new_impl = match slot_new {
        Some(slot_call) => quote! { #slot_call(cls, args, vm) },
        None => quote! { #base_class::slot_new(cls, args, vm) },
    };

    // We need this method, because of how `CPython` copies `__init__`
    // from `BaseException` in `SimpleExtendsException` macro.
    // See: `(initproc)BaseException_init`
    let init_method = match init {
        Some(init_def) => quote! { #init_def(zelf, args, vm) },
        None => quote! { #base_class::init(zelf, args, vm) },
    };

    let ret = quote! {
        #[pyexception(#class_name, #base_class)]
        #[derive(Debug)]
        #[doc = #docs]
        pub struct #class_name {}

        // We need this to make extend mechanism work:
        impl ::rustpython_vm::PyPayload for #class_name {
            fn class(vm: &::rustpython_vm::VirtualMachine) -> &'static ::rustpython_vm::Py<::rustpython_vm::builtins::PyType> {
                vm.ctx.exceptions.#ctx_name
            }
        }

        #[pyclass(flags(BASETYPE, HAS_DICT))]
        impl #class_name {
            #[pyslot]
            pub(crate) fn slot_new(
                cls: ::rustpython_vm::builtins::PyTypeRef,
                args: ::rustpython_vm::function::FuncArgs,
                vm: &::rustpython_vm::VirtualMachine,
            ) -> ::rustpython_vm::PyResult {
                #slot_new_impl
            }

            #[pyslot]
            #[pymethod(magic)]
            pub(crate) fn init(
                zelf: PyObjectRef,
                args: ::rustpython_vm::function::FuncArgs,
                vm: &::rustpython_vm::VirtualMachine,
            ) -> ::rustpython_vm::PyResult<()> {
                #init_method
            }
        }
    };
    Ok(ret)
}

/// #[pymethod] and #[pyclassmethod]
struct MethodItem {
    inner: ContentItemInner<AttrName>,
}

/// #[pyproperty]
struct GetSetItem {
    inner: ContentItemInner<AttrName>,
}

/// #[pyslot]
struct SlotItem {
    inner: ContentItemInner<AttrName>,
}

/// #[pyattr]
struct AttributeItem {
    inner: ContentItemInner<AttrName>,
}

/// #[extend_class]
struct ExtendClassItem {
    inner: ContentItemInner<AttrName>,
}

/// #[pymember]
struct MemberItem {
    inner: ContentItemInner<AttrName>,
}

impl ContentItem for MethodItem {
    type AttrName = AttrName;
    fn inner(&self) -> &ContentItemInner<AttrName> {
        &self.inner
    }
}
impl ContentItem for GetSetItem {
    type AttrName = AttrName;
    fn inner(&self) -> &ContentItemInner<AttrName> {
        &self.inner
    }
}
impl ContentItem for SlotItem {
    type AttrName = AttrName;
    fn inner(&self) -> &ContentItemInner<AttrName> {
        &self.inner
    }
}
impl ContentItem for AttributeItem {
    type AttrName = AttrName;
    fn inner(&self) -> &ContentItemInner<AttrName> {
        &self.inner
    }
}
impl ContentItem for ExtendClassItem {
    type AttrName = AttrName;
    fn inner(&self) -> &ContentItemInner<AttrName> {
        &self.inner
    }
}
impl ContentItem for MemberItem {
    type AttrName = AttrName;
    fn inner(&self) -> &ContentItemInner<AttrName> {
        &self.inner
    }
}

struct ImplItemArgs<'a, Item: ItemLike> {
    item: &'a Item,
    attrs: &'a mut Vec<Attribute>,
    context: &'a mut ImplContext,
    cfgs: &'a [Attribute],
}

trait ImplItem<Item>: ContentItem
where
    Item: ItemLike + ToTokens + GetIdent,
{
    fn gen_impl_item(&self, args: ImplItemArgs<'_, Item>) -> Result<()>;
}

impl<Item> ImplItem<Item> for MethodItem
where
    Item: ItemLike + ToTokens + GetIdent,
{
    fn gen_impl_item(&self, args: ImplItemArgs<'_, Item>) -> Result<()> {
        let func = args
            .item
            .function_or_method()
            .map_err(|_| self.new_syn_error(args.item.span(), "can only be on a method"))?;
        let ident = &func.sig().ident;

        let item_attr = args.attrs.remove(self.index());
        let item_meta = MethodItemMeta::from_attr(ident.clone(), &item_attr)?;

        let py_name = item_meta.method_name()?;
        let sig_doc = text_signature(func.sig(), &py_name);

        let tokens = {
            let doc = args.attrs.doc().map_or_else(TokenStream::new, |mut doc| {
                doc = format!("{}\n--\n\n{}", sig_doc, doc);
                quote!(.with_doc(#doc.to_owned(), ctx))
            });
            let build_func = match self.inner.attr_name {
                AttrName::Method => quote!(.build_method(ctx, class)),
                AttrName::ClassMethod => quote!(.build_classmethod(ctx, class)),
                AttrName::StaticMethod => quote!(.build_staticmethod(ctx, class)),
                other => unreachable!(
                    "Only 'method', 'classmethod' and 'staticmethod' are supported, got {:?}",
                    other
                ),
            };
            quote_spanned! { ident.span() =>
                class.set_str_attr(
                    #py_name,
                    ctx.make_funcdef(#py_name, Self::#ident)
                        #doc
                        #build_func,
                    ctx,
                );
            }
        };

        args.context.impl_extend_items.add_item(
            ident.clone(),
            vec![py_name],
            args.cfgs.to_vec(),
            tokens,
            5,
        )?;
        Ok(())
    }
}

impl<Item> ImplItem<Item> for GetSetItem
where
    Item: ItemLike + ToTokens + GetIdent,
{
    fn gen_impl_item(&self, args: ImplItemArgs<'_, Item>) -> Result<()> {
        let func = args
            .item
            .function_or_method()
            .map_err(|_| self.new_syn_error(args.item.span(), "can only be on a method"))?;
        let ident = &func.sig().ident;

        let item_attr = args.attrs.remove(self.index());
        let item_meta = GetSetItemMeta::from_attr(ident.clone(), &item_attr)?;

        let (py_name, kind) = item_meta.getset_name()?;
        args.context
            .getset_items
            .add_item(py_name, args.cfgs.to_vec(), kind, ident.clone())?;
        Ok(())
    }
}

impl<Item> ImplItem<Item> for SlotItem
where
    Item: ItemLike + ToTokens + GetIdent,
{
    fn gen_impl_item(&self, args: ImplItemArgs<'_, Item>) -> Result<()> {
        let (ident, span) = if let Ok(c) = args.item.constant() {
            (c.ident(), c.span())
        } else if let Ok(f) = args.item.function_or_method() {
            (&f.sig().ident, f.span())
        } else {
            return Err(self.new_syn_error(args.item.span(), "can only be on a method"));
        };

        let item_attr = args.attrs.remove(self.index());
        let item_meta = SlotItemMeta::from_attr(ident.clone(), &item_attr)?;

        let slot_ident = item_meta.slot_name()?;
        let slot_ident = Ident::new(&slot_ident.to_string().to_lowercase(), slot_ident.span());
        let slot_name = slot_ident.to_string();
        let tokens = {
            const NON_ATOMIC_SLOTS: &[&str] = &["as_buffer"];
            const POINTER_SLOTS: &[&str] = &["as_number"];
            if NON_ATOMIC_SLOTS.contains(&slot_name.as_str()) {
                quote_spanned! { span =>
                    slots.#slot_ident = Some(Self::#ident as _);
                }
            } else if POINTER_SLOTS.contains(&slot_name.as_str()) {
                quote_spanned! { span =>
                    slots.#slot_ident.store(Some(PointerSlot::from(Self::#ident())));
                }
            } else {
                quote_spanned! { span =>
                    slots.#slot_ident.store(Some(Self::#ident as _));
                }
            }
        };

        let pyname = format!("(slot {})", slot_name);
        args.context.extend_slots_items.add_item(
            ident.clone(),
            vec![pyname],
            args.cfgs.to_vec(),
            tokens,
            2,
        )?;

        Ok(())
    }
}

impl<Item> ImplItem<Item> for AttributeItem
where
    Item: ItemLike + ToTokens + GetIdent,
{
    fn gen_impl_item(&self, args: ImplItemArgs<'_, Item>) -> Result<()> {
        let cfgs = args.cfgs.to_vec();
        let attr = args.attrs.remove(self.index());

        let get_py_name = |attr: &Attribute, ident: &Ident| -> Result<_> {
            let item_meta = SimpleItemMeta::from_attr(ident.clone(), attr)?;
            let py_name = item_meta.simple_name()?;
            Ok(py_name)
        };
        let (ident, py_name, tokens) =
            if args.item.function_or_method().is_ok() || args.item.constant().is_ok() {
                let ident = args.item.get_ident().unwrap();
                let py_name = get_py_name(&attr, ident)?;

                let value = if args.item.constant().is_ok() {
                    // TODO: ctx.new_value
                    quote_spanned!(ident.span() => ctx.new_int(Self::#ident).into())
                } else {
                    quote_spanned!(ident.span() => Self::#ident(ctx))
                };
                (
                    ident,
                    py_name.clone(),
                    quote! {
                        class.set_str_attr(#py_name, #value, ctx);
                    },
                )
            } else {
                return Err(self.new_syn_error(
                    args.item.span(),
                    "can only be on a const or an associated method without argument",
                ));
            };

        args.context
            .impl_extend_items
            .add_item(ident.clone(), vec![py_name], cfgs, tokens, 1)?;

        Ok(())
    }
}

impl<Item> ImplItem<Item> for ExtendClassItem
where
    Item: ItemLike + ToTokens + GetIdent,
{
    fn gen_impl_item(&self, args: ImplItemArgs<'_, Item>) -> Result<()> {
        args.attrs.remove(self.index());

        let ident = &args
            .item
            .function_or_method()
            .map_err(|_| self.new_syn_error(args.item.span(), "can only be on a method"))?
            .sig()
            .ident;

        args.context.class_extensions.push(quote! {
            Self::#ident(ctx, class);
        });

        Ok(())
    }
}

impl<Item> ImplItem<Item> for MemberItem
where
    Item: ItemLike + ToTokens + GetIdent,
{
    fn gen_impl_item(&self, args: ImplItemArgs<'_, Item>) -> Result<()> {
        let func = args
            .item
            .function_or_method()
            .map_err(|_| self.new_syn_error(args.item.span(), "can only be on a method"))?;
        let ident = &func.sig().ident;

        let item_attr = args.attrs.remove(self.index());
        let item_meta = MemberItemMeta::from_attr(ident.clone(), &item_attr)?;

        let py_name = item_meta.member_name()?;
        let tokens = {
            quote_spanned! { ident.span() =>
                class.set_str_attr(
                    #py_name,
                    ctx.new_member(#py_name, Self::#ident, class),
                    ctx,
                );
            }
        };

        args.context.impl_extend_items.add_item(
            ident.clone(),
            vec![py_name],
            args.cfgs.to_vec(),
            tokens,
            5,
        )?;
        Ok(())
    }
}

#[derive(Default)]
#[allow(clippy::type_complexity)]
struct GetSetNursery {
    map: HashMap<(String, Vec<Attribute>), (Option<Ident>, Option<Ident>, Option<Ident>)>,
    validated: bool,
}

enum GetSetItemKind {
    Get,
    Set,
    Delete,
}

impl GetSetNursery {
    fn add_item(
        &mut self,
        name: String,
        cfgs: Vec<Attribute>,
        kind: GetSetItemKind,
        item_ident: Ident,
    ) -> Result<()> {
        assert!(!self.validated, "new item is not allowed after validation");
        if !matches!(kind, GetSetItemKind::Get) && !cfgs.is_empty() {
            return Err(syn::Error::new_spanned(
                item_ident,
                "Only the getter can have #[cfg]",
            ));
        }
        let entry = self.map.entry((name.clone(), cfgs)).or_default();
        let func = match kind {
            GetSetItemKind::Get => &mut entry.0,
            GetSetItemKind::Set => &mut entry.1,
            GetSetItemKind::Delete => &mut entry.2,
        };
        if func.is_some() {
            return Err(syn::Error::new_spanned(
                item_ident,
                format!("Multiple property accessors with name '{}'", name),
            ));
        }
        *func = Some(item_ident);
        Ok(())
    }

    fn validate(&mut self) -> Result<()> {
        let mut errors = Vec::new();
        for ((name, _cfgs), (getter, setter, deleter)) in &self.map {
            if getter.is_none() {
                errors.push(syn::Error::new_spanned(
                    setter.as_ref().or(deleter.as_ref()).unwrap(),
                    format!("GetSet '{}' is missing a getter", name),
                ));
            };
        }
        errors.into_result()?;
        self.validated = true;
        Ok(())
    }
}

impl ToTokens for GetSetNursery {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        assert!(self.validated, "Call `validate()` before token generation");
        let properties = self
            .map
            .iter()
            .map(|((name, cfgs), (getter, setter, deleter))| {
                let setter = match setter {
                    Some(setter) => quote_spanned! { setter.span() => .with_set(&Self::#setter)},
                    None => quote! {},
                };
                let deleter = match deleter {
                    Some(deleter) => {
                        quote_spanned! { deleter.span() => .with_delete(&Self::#deleter)}
                    }
                    None => quote! {},
                };
                quote_spanned! { getter.span() =>
                    #( #cfgs )*
                    class.set_str_attr(
                        #name,
                        ::rustpython_vm::PyRef::new_ref(
                            ::rustpython_vm::builtins::PyGetSet::new(#name.into(), class)
                                .with_get(&Self::#getter)
                                #setter #deleter,
                                ctx.types.getset_type.to_owned(), None),
                        ctx
                    );
                }
            });
        tokens.extend(properties);
    }
}

struct MethodItemMeta(ItemMetaInner);

impl ItemMeta for MethodItemMeta {
    const ALLOWED_NAMES: &'static [&'static str] = &["name", "magic"];

    fn from_inner(inner: ItemMetaInner) -> Self {
        Self(inner)
    }
    fn inner(&self) -> &ItemMetaInner {
        &self.0
    }
}

impl MethodItemMeta {
    fn method_name(&self) -> Result<String> {
        let inner = self.inner();
        let name = inner._optional_str("name")?;
        let magic = inner._bool("magic")?;
        Ok(if let Some(name) = name {
            name
        } else {
            let name = inner.item_name();
            if magic {
                format!("__{}__", name)
            } else {
                name
            }
        })
    }
}

struct GetSetItemMeta(ItemMetaInner);

impl ItemMeta for GetSetItemMeta {
    const ALLOWED_NAMES: &'static [&'static str] = &["name", "magic", "setter", "deleter"];

    fn from_inner(inner: ItemMetaInner) -> Self {
        Self(inner)
    }
    fn inner(&self) -> &ItemMetaInner {
        &self.0
    }
}

impl GetSetItemMeta {
    fn getset_name(&self) -> Result<(String, GetSetItemKind)> {
        let inner = self.inner();
        let magic = inner._bool("magic")?;
        let kind = match (inner._bool("setter")?, inner._bool("deleter")?) {
            (false, false) => GetSetItemKind::Get,
            (true, false) => GetSetItemKind::Set,
            (false, true) => GetSetItemKind::Delete,
            (true, true) => {
                return Err(syn::Error::new_spanned(
                    &inner.meta_ident,
                    format!(
                        "can't have both setter and deleter on a #[{}] fn",
                        inner.meta_name()
                    ),
                ))
            }
        };
        let name = inner._optional_str("name")?;
        let py_name = if let Some(name) = name {
            name
        } else {
            let sig_name = inner.item_name();
            let extract_prefix_name = |prefix, item_typ| {
                if let Some(name) = sig_name.strip_prefix(prefix) {
                    if name.is_empty() {
                        Err(syn::Error::new_spanned(
                            &inner.meta_ident,
                            format!(
                                "A #[{}({typ})] fn with a {prefix}* name must \
                                 have something after \"{prefix}\"",
                                inner.meta_name(),
                                typ = item_typ,
                                prefix = prefix
                            ),
                        ))
                    } else {
                        Ok(name.to_owned())
                    }
                } else {
                    Err(syn::Error::new_spanned(
                        &inner.meta_ident,
                        format!(
                            "A #[{}(setter)] fn must either have a `name` \
                             parameter or a fn name along the lines of \"set_*\"",
                            inner.meta_name()
                        ),
                    ))
                }
            };
            let name = match kind {
                GetSetItemKind::Get => sig_name,
                GetSetItemKind::Set => extract_prefix_name("set_", "setter")?,
                GetSetItemKind::Delete => extract_prefix_name("del_", "deleter")?,
            };
            if magic {
                format!("__{}__", name)
            } else {
                name
            }
        };
        Ok((py_name, kind))
    }
}

struct SlotItemMeta(ItemMetaInner);

impl ItemMeta for SlotItemMeta {
    const ALLOWED_NAMES: &'static [&'static str] = &[]; // not used

    fn from_nested<I>(item_ident: Ident, meta_ident: Ident, mut nested: I) -> Result<Self>
    where
        I: std::iter::Iterator<Item = NestedMeta>,
    {
        let meta_map = if let Some(nmeta) = nested.next() {
            match nmeta {
                NestedMeta::Meta(meta) => {
                    Some([("name".to_owned(), (0, meta))].iter().cloned().collect())
                }
                _ => None,
            }
        } else {
            Some(HashMap::default())
        };
        match (meta_map, nested.next()) {
            (Some(meta_map), None) => Ok(Self::from_inner(ItemMetaInner {
                item_ident,
                meta_ident,
                meta_map,
            })),
            _ => Err(syn::Error::new_spanned(
                meta_ident,
                "#[pyslot] must be of the form #[pyslot] or #[pyslot(slotname)]",
            )),
        }
    }

    fn from_inner(inner: ItemMetaInner) -> Self {
        Self(inner)
    }
    fn inner(&self) -> &ItemMetaInner {
        &self.0
    }
}

impl SlotItemMeta {
    fn slot_name(&self) -> Result<Ident> {
        let inner = self.inner();
        let slot_name = if let Some((_, meta)) = inner.meta_map.get("name") {
            match meta {
                Meta::Path(path) => path.get_ident().cloned(),
                _ => None,
            }
        } else {
            let ident_str = self.inner().item_name();
            let name = if let Some(stripped) = ident_str.strip_prefix("slot_") {
                proc_macro2::Ident::new(stripped, inner.item_ident.span())
            } else {
                inner.item_ident.clone()
            };
            Some(name)
        };
        slot_name.ok_or_else(|| {
            syn::Error::new_spanned(
                &inner.meta_ident,
                "#[pyslot] must be of the form #[pyslot] or #[pyslot(slotname)]",
            )
        })
    }
}

struct MemberItemMeta(ItemMetaInner);

impl ItemMeta for MemberItemMeta {
    const ALLOWED_NAMES: &'static [&'static str] = &["magic"];

    fn from_inner(inner: ItemMetaInner) -> Self {
        Self(inner)
    }
    fn inner(&self) -> &ItemMetaInner {
        &self.0
    }
}

impl MemberItemMeta {
    fn member_name(&self) -> Result<String> {
        let inner = self.inner();
        let magic = inner._bool("magic")?;
        let name = inner.item_name();
        Ok(if magic { format!("__{}__", name) } else { name })
    }
}

struct ExtractedImplAttrs {
    with_impl: TokenStream,
    with_slots: TokenStream,
    flags: TokenStream,
}

fn extract_impl_attrs(attr: AttributeArgs, item: &Ident) -> Result<ExtractedImplAttrs> {
    let mut withs = Vec::new();
    let mut with_slots = Vec::new();
    let mut flags = vec![quote! {
        {
            #[cfg(not(debug_assertions))] {
                ::rustpython_vm::types::PyTypeFlags::DEFAULT
            }
            #[cfg(debug_assertions)] {
                ::rustpython_vm::types::PyTypeFlags::DEFAULT
                    .union(::rustpython_vm::types::PyTypeFlags::_CREATED_WITH_FLAGS)
            }
        }
    }];

    for attr in attr {
        match attr {
            NestedMeta::Meta(Meta::List(syn::MetaList { path, nested, .. })) => {
                if path_eq(&path, "with") {
                    for meta in nested {
                        let path = match meta {
                            NestedMeta::Meta(Meta::Path(path)) => path,
                            meta => {
                                bail_span!(meta, "#[pyclass(with(...))] arguments should be paths")
                            }
                        };
                        let (extend_class, extend_slots) = if path_eq(&path, "PyRef") {
                            // special handling for PyRef
                            (
                                quote!(PyRef::<Self>::impl_extend_class),
                                quote!(PyRef::<Self>::extend_slots),
                            )
                        } else {
                            (
                                quote!(<Self as #path>::__extend_py_class),
                                quote!(<Self as #path>::__extend_slots),
                            )
                        };
                        let item_span = item.span().resolved_at(Span::call_site());
                        withs.push(quote_spanned! { path.span() =>
                            #extend_class(ctx, class);
                        });
                        with_slots.push(quote_spanned! { item_span =>
                            #extend_slots(slots);
                        });
                    }
                } else if path_eq(&path, "flags") {
                    for meta in nested {
                        match meta {
                            NestedMeta::Meta(Meta::Path(path)) => {
                                if let Some(ident) = path.get_ident() {
                                    flags.push(quote_spanned! { ident.span() =>
                                         .union(::rustpython_vm::types::PyTypeFlags::#ident)
                                    });
                                } else {
                                    bail_span!(
                                        path,
                                        "#[pyclass(flags(...))] arguments should be ident"
                                    )
                                }
                            }
                            meta => {
                                bail_span!(meta, "#[pyclass(flags(...))] arguments should be ident")
                            }
                        }
                    }
                } else {
                    bail_span!(path, "Unknown pyimpl attribute")
                }
            }
            attr => bail_span!(attr, "Unknown pyimpl attribute"),
        }
    }

    Ok(ExtractedImplAttrs {
        with_impl: quote! {
            #(#withs)*
        },
        flags: quote! {
            #(#flags)*
        },
        with_slots: quote! {
            #(#with_slots)*
        },
    })
}

fn impl_item_new<Item>(
    index: usize,
    attr_name: AttrName,
) -> Result<Box<dyn ImplItem<Item, AttrName = AttrName>>>
where
    Item: ItemLike + ToTokens + GetIdent,
{
    use AttrName::*;
    Ok(match attr_name {
        attr_name @ Method | attr_name @ ClassMethod | attr_name @ StaticMethod => {
            Box::new(MethodItem {
                inner: ContentItemInner { index, attr_name },
            })
        }
        GetSet => Box::new(GetSetItem {
            inner: ContentItemInner { index, attr_name },
        }),
        Slot => Box::new(SlotItem {
            inner: ContentItemInner { index, attr_name },
        }),
        Attr => Box::new(AttributeItem {
            inner: ContentItemInner { index, attr_name },
        }),
        ExtendClass => Box::new(ExtendClassItem {
            inner: ContentItemInner { index, attr_name },
        }),
        Member => Box::new(MemberItem {
            inner: ContentItemInner { index, attr_name },
        }),
    })
}

fn attrs_to_content_items<F, R>(
    attrs: &[Attribute],
    item_new: F,
) -> Result<(Vec<R>, Vec<Attribute>)>
where
    F: Fn(usize, AttrName) -> Result<R>,
{
    let mut cfgs: Vec<Attribute> = Vec::new();
    let mut result = Vec::new();

    let mut iter = attrs.iter().enumerate().peekable();
    while let Some((_, attr)) = iter.peek() {
        // take all cfgs but no py items
        let attr = *attr;
        let attr_name = if let Some(ident) = attr.get_ident() {
            ident.to_string()
        } else {
            continue;
        };
        if attr_name == "cfg" {
            cfgs.push(attr.clone());
        } else if ALL_ALLOWED_NAMES.contains(&attr_name.as_str()) {
            break;
        }
        iter.next();
    }

    for (i, attr) in iter {
        // take py items but no cfgs
        let attr_name = if let Some(ident) = attr.get_ident() {
            ident.to_string()
        } else {
            continue;
        };
        if attr_name == "cfg" {
            return Err(syn::Error::new_spanned(
                attr,
                "#[py*] items must be placed under `cfgs`",
            ));
        }
        let attr_name = match AttrName::from_str(attr_name.as_str()) {
            Ok(name) => name,
            Err(wrong_name) => {
                if ALL_ALLOWED_NAMES.contains(&attr_name.as_str()) {
                    return Err(syn::Error::new_spanned(
                        attr,
                        format!("#[pyclass] doesn't accept #[{}]", wrong_name),
                    ));
                } else {
                    continue;
                }
            }
        };

        result.push(item_new(i, attr_name)?);
    }
    Ok((result, cfgs))
}

#[derive(Debug)]
pub(crate) struct PyExceptionDef {
    pub class_name: Ident,
    pub base_class: Ident,
    pub ctx_name: Ident,
    pub docs: LitStr,

    /// Holds optional `slot_new` slot to be used instead of a default one:
    pub slot_new: Option<Ident>,
    /// We also store `__init__` magic method, that can
    pub init: Option<Ident>,
}

impl Parse for PyExceptionDef {
    fn parse(input: ParseStream) -> ParsingResult<Self> {
        let class_name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;

        let base_class: Ident = input.parse()?;
        input.parse::<Token![,]>()?;

        let ctx_name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;

        let docs: LitStr = input.parse()?;
        input.parse::<Option<Token![,]>>()?;

        let slot_new: Option<Ident> = input.parse()?;
        input.parse::<Option<Token![,]>>()?;

        let init: Option<Ident> = input.parse()?;
        input.parse::<Option<Token![,]>>()?; // leading `,`

        Ok(PyExceptionDef {
            class_name,
            base_class,
            ctx_name,
            docs,
            slot_new,
            init,
        })
    }
}

fn parse_vec_ident(
    attr: &[NestedMeta],
    item: &Item,
    index: usize,
    message: &str,
) -> Result<String> {
    Ok(attr
        .get(index)
        .ok_or_else(|| {
            syn::Error::new_spanned(&item, format!("We require {} argument to be set", &message))
        })?
        .get_ident()
        .ok_or_else(|| {
            syn::Error::new_spanned(
                &item,
                format!("We require {} argument to be ident or string", &message),
            )
        })?
        .to_string())
}
