import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createYoga } from 'graphql-yoga';
import { describe, expect, it } from 'vitest';

import { loadShopData } from '../data/loader.js';
import { type SandboxSchemaResolvers, createSandboxSchema } from '../schema.js';
import { CartStore } from './cart.js';
import { homepageResolvers } from './homepage.js';
import type { ResolverContext } from './index.js';

const FIXTURE_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../tests/fixtures/sandbox_shop_v0',
);

const BASE_URL = 'https://shop.example';

const data = loadShopData(FIXTURE_DIR);
const carts = new CartStore();

const resolvers: SandboxSchemaResolvers = {
  Query: homepageResolvers.Query,
};

const yoga = createYoga({
  schema: createSandboxSchema(resolvers),
  context: (): ResolverContext => ({ data, carts, baseUrl: BASE_URL }),
});

interface ExecutionResult {
  readonly data?: unknown;
  readonly errors?: readonly unknown[];
}

async function run(source: string): Promise<ExecutionResult> {
  const response = await yoga.fetch(`${BASE_URL}/graphql`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ query: source }),
  });
  return (await response.json()) as ExecutionResult;
}

describe('homepageResolvers — Query.homepage', () => {
  it('returns the hero and banners with hotlinked absolute CDN urls', async () => {
    const result = await run(/* GraphQL */ `
      {
        homepage {
          hero {
            url
            width
            height
          }
          banners {
            url
            width
            height
          }
        }
      }
    `);
    expect(result.errors).toBeUndefined();
    expect(result.data).toEqual({
      homepage: {
        hero: { url: 'https://cdn.example.com/hero-spring.jpg', width: 2400, height: 1200 },
        banners: [{ url: 'https://cdn.example.com/promo-sale.jpg', width: 1600, height: 600 }],
      },
    });
  });
});
