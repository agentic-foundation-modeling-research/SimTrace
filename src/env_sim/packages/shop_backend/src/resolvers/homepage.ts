/**
 * Resolver for the site-level `Query.homepage` field.
 *
 * `homepage` exposes the editorial imagery (a single hero plus an ordered list
 * of promo banners) reused verbatim from the original storefront. The images
 * live in the optional `homepage.json` dataset file; when that file is absent
 * the loader defaults to `{ hero: null, banners: [] }`, so this resolver
 * naturally returns an empty hero and no banners.
 *
 * Each image is materialized through the shared `buildImageNode`, so hotlinked
 * absolute CDN `src` values pass through `rewriteImageUrl` untouched.
 */

import { type ImageNode, buildImageNode } from './builders.js';
import type { ResolverContext } from './index.js';

/** `Query.homepage` node shape. Non-null: the field always resolves. */
export interface HomepageNode {
  readonly hero: ImageNode | null;
  readonly banners: readonly ImageNode[];
}

/**
 * Resolver map for the homepage area. Wired into the combined resolver map in
 * `index.ts`.
 */
export const homepageResolvers = {
  Query: {
    homepage: (_parent: unknown, _args: unknown, ctx: ResolverContext): HomepageNode => {
      const { hero, banners } = ctx.data.homepage;
      return {
        hero: hero === null ? null : buildImageNode(hero, ctx.baseUrl),
        banners: banners.map((banner) => buildImageNode(banner, ctx.baseUrl)),
      };
    },
  },
};
