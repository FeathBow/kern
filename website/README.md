# Kern website

Static product page for Kern: models ship as verified GPU programs.

## Local development

Requires Node.js 18 or newer.

```sh
cd website
npm ci
npm run dev
```

Create and inspect the production bundle:

```sh
npm run build
npm run preview
```

The production output is written to `dist/`.

## Cloudflare Pages

The site can deploy directly from this repository without a Worker:

| Setting | Value |
| --- | --- |
| Repository | `pegainfer-project/kern` |
| Production branch | `master` |
| Root directory | `website` |
| Build command | `npm run build` |
| Build output directory | `dist` |

In the Cloudflare dashboard, create a Pages application, import the GitHub
repository, and enter the settings above. Pages will build from `website/` and
publish new commits to `master` automatically. Pull requests receive preview
deployments.

## Content boundaries

- Product structure and counts come from the checked-in Qwen3-4B manifests.
- Performance figures reproduce repository measurements and keep their test
  conditions visible.
- The TileFoundry section describes conceptual alignment. There is no direct
  TileFoundry-to-Kern exporter today.
- The `<3K` source metric counts production Rust files in `kern-manifest` and
  `kern-runtime`, including `kern-run`; it excludes tests and tools.
