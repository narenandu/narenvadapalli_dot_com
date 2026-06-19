// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import mdx from '@astrojs/mdx';

// A simple Remark plugin to handle youtube embeds and custom shortcodes
function remarkCustomFeatures() {
  return (tree) => {
    function extractYoutubeId(url) {
      const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
      const match = url.match(regExp);
      return (match && match[2].length === 11) ? match[2] : null;
    }

    function visit(node) {
      // 1. YouTube embeds inside inline code `youtube: URL`
      if (node.type === 'inlineCode' && node.value.startsWith('youtube:')) {
        const url = node.value.replace('youtube:', '').trim();
        const id = extractYoutubeId(url);
        if (id) {
          node.type = 'html';
          node.value = `<div class="video-wrapper"><iframe src="https://www.youtube.com/embed/${id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>`;
        }
      }

      // 2. Shortcodes [row], [col], [/row], [/col]
      if (node.type === 'text' && (node.value.includes('[row]') || node.value.includes('[col]') || node.value.includes('[/row]') || node.value.includes('[/col]'))) {
        let val = node.value;
        val = val.replaceAll('[row]', '<div class="grid-row">');
        val = val.replaceAll('[/row]', '</div>');
        val = val.replaceAll('[col]', '<div class="portfolio-item-col">');
        val = val.replaceAll('[/col]', '</div>');
        node.type = 'html';
        node.value = val;
      }

      if (node.children) {
        node.children.forEach(visit);
      }
    }

    visit(tree);
  };
}

// https://astro.build/config
export default defineConfig({
  integrations: [react(), mdx()],
  markdown: {
    remarkPlugins: [remarkCustomFeatures]
  }
});