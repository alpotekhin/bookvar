import { defineCollection, z } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema({extend: z.object({
    sourceLanguage: z.enum(['en', 'ru', 'mixed']).optional(),
    translationStatus: z.enum(['current', 'partial', 'canonical']).optional(),
    canonicalHref: z.string().optional(),
    translationReviewed: z.string().optional(),
  })}) })
};
