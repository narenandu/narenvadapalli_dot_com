import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blog = defineCollection({
  loader: glob({ pattern: '**/index.md', base: './contents/blog' }),
  schema: ({ image }) => z.object({
    title: z.string(),
    date: z.coerce.date(),
    template: z.string().optional(),
    image: image().optional(),
    banner: z.string().optional(),
    description: z.string().optional(),
    tags: z.array(z.string()).optional(),
  }),
});

const portfolio = defineCollection({
  loader: glob({ pattern: '**/index.md', base: './contents/portfolio' }),
  schema: ({ image }) => z.object({
    title: z.string(),
    date: z.coerce.date(),
    template: z.string().optional(),
    image: image().optional(),
    description: z.string().optional(),
  }),
});

const basepages = defineCollection({
  loader: glob({ pattern: '**/index.md', base: './contents/basepages' }),
  schema: ({ image }) => z.object({
    title: z.string().optional(),
    description: z.string().optional(),
    date: z.coerce.date().optional(),
    template: z.string().optional(),
    image: image().optional(),
  }),
});

export const collections = { blog, portfolio, basepages };
