# Narendra Kumar Vadapalli - Personal Website & Blog

A high-performance, responsive personal website and blog built with **Astro 5+**, **React**, **MDX**, and a custom **Vanilla CSS** design system. 

It is designed to be fully static, fast (zero-JS by default), and deployed continuously via AWS Amplify Hosting.

---

## 🚀 Tech Stack & Features

- **Framework**: [Astro 5+](https://astro.build/) (Static Site Generation mode)
- **Component Support**: [React 19](https://react.dev/) (via Astro React Integration for island hydration if needed)
- **Content Format**: Markdown (`.md`) and MDX (`.mdx`) with native type-safe **Astro Content Collections** and frontmatter schema validation via Zod.
- **Styling**: Vanilla CSS (highly modular, clean variables, custom glassmorphism components, custom timeline).
- **SEO & Analytics**: Automatic OpenGraph tags, semantic HTML, responsive metadata, and Google Analytics (GA4) tracking.
- **Continuous Deployment**: Continuous push-to-deploy hosting on AWS Amplify.

---

## 📂 Project Structure

- `contents/` - Markdown content files parsed via Content Collections.
  - `blog/` - Technical blog posts. Each post is housed in its own directory with images.
  - `portfolio/` - Portfolio showcase items (e.g. Media Credits, Graduation Work).
  - `basepages/` - Basic static page markdown files (e.g. About page content).
- `public/` - Static assets copied directly to the build root (images, logos, ads.txt, search console verification files).
- `src/` - Application source code.
  - `components/` - Reusable UI widgets and interactive components.
  - `layouts/` - Master layout templates (e.g., `Layout.astro`).
  - `pages/` - Route definitions (Astro file-based routing).
  - `styles/` - Global styling configurations (`global.css`).
- `astro.config.mjs` - Integrations setup and custom Remark plugins.

---

## 🛠️ Local Development

### 1. Prerequisites
Ensure you have **Node.js >= 22.12.0** installed. (Recommended: Node 23).

### 2. Install Dependencies
```bash
npm install
```

### 3. Start Development Server
```bash
npm run dev
```
Open **[http://localhost:4321](http://localhost:4321)** in your browser. The server supports hot-module reloading.

### 4. Build Production Bundle
To compile and test the static build output:
```bash
npm run build
```
The optimized HTML, CSS, JavaScript, and WebP images are compiled into the `dist/` directory.

---

## ⚙️ Deployment (AWS Amplify)

The app is continuously built and hosted on AWS Amplify. The build configurations are managed in `amplify.yml`:
- Installs Node.js 22+ using `nvm`.
- Installs packages via `npm ci`.
- Compiles the static build via `npm run build`.
- Serves the static outputs from the `dist/` folder.
