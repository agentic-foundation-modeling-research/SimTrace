# Fixes for `gen_homepage`

These rules supersede anything in `execute.md` §3 they contradict.

## Hide data-driven sections when their backing resource is absent

A SandboxShop only ships the data its source merchant actually
exposed. If you author a homepage section whose URLs depend on a
specific resource — most often a blog (`/blogs/recipes`,
`/blogs/journal`) but also custom pages, lookbooks, or reviews — and
that resource is missing from the dataset, every link the section
renders becomes a dead 404. Empty-state placeholders ("Recipes are
on the way → /blogs/recipes") are *worse* than no section, because
they advertise the broken link.

Gate the section render on the **resource itself**, not on its
populated children. For a blog-backed carousel:

```tsx
{blogs.recipes?.blog ? (
  <RecipeCarousel
    blogHandle={RECIPE_BLOG_HANDLE}
    articles={blogs.recipes.blog.articles.nodes}
  />
) : null}
```

Checking `articles.length > 0` is not enough — a real blog with zero
published articles is a valid empty state and should still render
the "See all" link. A *missing* blog (the storefront query returned
`{blog: null}`) must drop the entire section. Do the same for any
section that links into a resource you cannot guarantee exists in
every shop's dataset.

## Align section heads and bodies to the same container

A homepage section typically has two parts: a **head row** (`<h2>` +
optional "Shop all →" link, often using `.home-section-head`) and a
**body** (a grid, scroller, or banner). On wide viewports it is easy
to land in a state where one part is constrained to the page
container (`max-width: var(--container-max); margin-inline: auto`)
while the other is full-bleed — the two then visibly drift apart at
1440px+ and read as misaligned.

When you author or edit a homepage section, make a deliberate choice
*and apply it to both parts*:

- **Constrained** (the default): wrap **both** the head row and the
  body in `.container`, or apply `max-width: var(--container-max);
  margin-inline: auto` to whichever wrapper they share. Horizontal
  carousels/scrollers (`.h-scroller` / `HorizontalScroller`) belong
  in this bucket — their inline padding alone does not center them.
- **Full-bleed** (a deliberate hero / banner): both parts span the
  viewport. In that case the head row should also drop its
  `.container` wrapper or include explicit edge padding so the
  heading does not float in the middle of an otherwise edge-to-edge
  section.

If only one part inherits the container constraint, the heading row
left/right edges stop matching the body's left/right edges and the
section reads as broken. `tsc` and `build` will not catch this; only
a wide-viewport visual pass will.

## Use the shop's own imagery for image-bearing homepage sections

The SandboxShop dataset exposes the source store's real editorial
imagery through dedicated Storefront API fields, plus product image
records and (on AI-image runs) generated PNGs under `data/images/`.
Homepage sections that are visually image-bearing must bind to those
fields instead of CSS-only placeholders.

Prefer the sanctioned fields in this order:

- **Hero media**: query the root `homepage { hero { id url altText
  width height } }` field and render `homepage.hero` as the homepage
  hero. It is the source store's real above-the-fold image.
- **Promo banners**: query `homepage { banners { id url altText width
  height } }` and render `homepage.banners` in order for any promo /
  `image_with_text` banner strip.
- **Collection / category cards**: query `collection.image { id url
  altText width height }`. For collections whose `image` is `null`,
  fall back to the first product's `featuredImage`.
- **Product cards**: product `featuredImage { id url altText width
  height }`.

Render with a plain `<img>` (or Hydrogen's `<Image data={...}>` when
present) using the returned `url`; keep a fixed aspect ratio on the
container so lazy loading cannot collapse the layout.

These `url` values may resolve to absolute CDN addresses — that is
expected and correct: the hero and banners are reused verbatim from the
source store, so binding these fields is the *required* data path. The
standing rule against **inventing or hard-coding** image URLs still
holds — never paste a literal external URL or a stock-photo link into
the source. Bind the GraphQL field; do not fabricate the address.

When `homepage.hero` (or `homepage.banners`) is `null`/empty, the
capture did not run for this shop — render the section's image-less
layout rather than improvising a hero from a product cutout.

Do not satisfy an image slot with only a gradient block, initials,
`role="img"` text badge, "Portrait — ..." label, or other CSS-drawn
placeholder when real imagery is available. Those can be decorative
overlays, but they cannot be the only media in a section whose manual
describes image/text or product/category imagery.
