# Garza Lab website

Live site: https://garza-lab.netlify.app

## Layout

This folder is the deployed site. Upload the whole folder (drag it onto Netlify Drop,
or push it to the repo root).

- `index.html` — the site. Edit this file for text changes.
- `support.js` — runtime the page loads.
- `_ds/` — stylesheet and component bundle.
- `uploads/opt/` — optimized figures and photos.

There is no build step and no load splash: the page paints as it loads, and each block
fades in as it scrolls into view.

## Netlify build settings

- Build command: *(leave empty)*
- Publish directory: `.`

## Assets still hosted elsewhere

A few team photos, the two journal covers and the PAYS group photo load from
`garzalab.weebly.com`. Copy them into `uploads/opt/` and repoint the `src` attributes in
`index.html` before that site is taken down.
