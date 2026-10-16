# Notebook website

In Settings → Pages, choose **GitHub Actions** as the source. Push to `main`
or run **Publish notebook website** manually. No GitHub App, PAT or custom
secret is required. Organization policies and environment rules still apply.

Source: `3dglauber-bayes-example/mcmc`. The site preserves its notebook folder
hierarchy and renders saved outputs without executing notebook code. Full
builds prevent stale pages when notebooks are renamed or deleted.

Edit `website.yml` for home text. Edit `template/` for layout and styles.
The template and renderer were copied from `naavilam/github-infra-templates`;
updates there are not applied automatically. The author biography and personal
contact were removed, the upper-left icon is noninteractive, and Analytics
was omitted. Other template pages and navigation are preserved.

Local build (after installing requirements.txt):

```sh
python .github/pages/build_site.py --src 3dglauber-bayes-example/mcmc --out /tmp/glauber-site --template .github/pages/template --cfg .github/pages/website.yml --title "3D Glauber — Bayesian Analysis" --execute false
```

Only repositories containing this workflow publish a site. To reuse it elsewhere,
copy `.github/pages` and the workflow, then change the source folder and metadata.
