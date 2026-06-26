/**
 * pieces.js — self-contained SVG chess piece renderer
 *
 * Returns data URI strings for use as chessboard.js pieceTheme.
 * No external images needed — pieces are rendered from SVG paths.
 * Based on the Colin M.L. Burnett Wikipedia piece set (CC BY-SA 3.0).
 */

'use strict';

// SVG path data for each piece type (simplified Burnett-style paths)
const _PATHS = {
  // ── White pieces ──────────────────────────────────────────────
  wK: `<g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M22.5 11.63V6M20 8h5" stroke-linejoin="miter"/>
    <path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#fff" stroke-linecap="butt" stroke-linejoin="miter"/>
    <path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V17s.5-5.5-5.5-5.5S11.5 17 11.5 17v6.5c-2.5-7.5-12-10.5-16-4-3 6 5 10 5 10V37z" fill="#fff"/>
    <path d="M11.5 30c5.5-3 15.5-3 21 0M11.5 33.5c5.5-3 15.5-3 21 0M11.5 37c5.5-3 15.5-3 21 0"/>
  </g>`,
  wQ: `<g fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M8 12a2 2 0 1 1-4 0 2 2 0 0 1 4 0zM24.5 7.5a2 2 0 1 1-4 0 2 2 0 0 1 4 0zM41 12a2 2 0 1 1-4 0 2 2 0 0 1 4 0zM16 8.5a2 2 0 1 1-4 0 2 2 0 0 1 4 0zM33 8.5a2 2 0 1 1-4 0 2 2 0 0 1 4 0z"/>
    <path d="M9 26c8.5-8.5 15-6.5 23 0M9 26c8.5-13.5 28-13.5 30 0" stroke-linecap="butt"/>
    <path d="M11.5 30c6-5 15-5 21 0" stroke-linecap="butt"/>
    <path d="M11 38.5c5.5-3.5 15.5-3.5 21 0V36c-6-3-15-3-21 0v2.5z"/>
    <path d="M11 29.5c6-4.5 15-4.5 21 0V26l-3 1-3.5-4.5-4 5-4-5L14 27l-3-1v3.5z" stroke-linecap="butt" stroke-linejoin="miter"/>
    <path d="M11 38.5c0-3.5 3-5 9-5h3c6 0 9 1.5 9 5" stroke-linecap="butt"/>
  </g>`,
  wR: `<g fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M9 39h27v-3H9v3zM12 36v-4h21v4H12zM11 14V9h4v2h5V9h5v2h5V9h4v5" stroke-linecap="butt" stroke-linejoin="miter"/>
    <path d="M34 14l-3 3H14l-3-3"/>
    <path d="M31 17v12.5H14V17" stroke-linecap="butt" stroke-linejoin="miter"/>
    <path d="M31 29.5l1.5 2.5h-20l1.5-2.5"/>
    <path d="M11 14h23" fill="none" stroke-linejoin="miter"/>
  </g>`,
  wB: `<g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <g fill="#fff" stroke-linecap="butt">
      <path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2z"/>
      <path d="M15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z"/>
      <path d="M25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z"/>
    </g>
    <path d="M17.5 26h10M15 30h15M22.5 15.5v5M20 18h5" stroke-linejoin="miter"/>
  </g>`,
  wN: `<g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21" fill="#fff"/>
    <path d="M24 18c.38 5.12-1.5 7-7.5 7.5c4 3 7.2 1 11 5.5c-3 3.5-1 8.5 4 9.5" fill="#fff"/>
    <path d="M9.5 25.5a.5.5 0 1 1-1 0 .5.5 0 0 1 1 0z" fill="#000" stroke="#000"/>
    <path d="M14.933 15.75a1.75 1.75 0 1 1-3.5 0 1.75 1.75 0 0 1 3.5 0z" fill="#000" stroke="#000" stroke-width="1"/>
    <path d="M20.5 12c0 1 1 2 1 2v3l-1.5.5c-1.5 0-3.5.5-5 2s-3 3.5-2.5 5c.5 1.5 2.5.5 2.5.5" fill="#fff"/>
    <path d="M12.5 20c-1.5 1-4.5 1-4.5 3 0 1 1.5 2 4 2.5" fill="#fff" stroke-linejoin="round"/>
  </g>`,
  wP: `<g fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03C15.41 27.09 11 31.58 11 39.5H34c0-7.92-4.41-12.41-7.41-13.47C28.06 24.84 29 23.03 29 21c0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z"/>
  </g>`,

  // ── Black pieces ──────────────────────────────────────────────
  bK: `<g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M22.5 11.63V6" stroke-linejoin="miter"/>
    <path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#000" stroke-linecap="butt" stroke-linejoin="miter"/>
    <path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V17s.5-5.5-5.5-5.5S11.5 17 11.5 17v6.5c-2.5-7.5-12-10.5-16-4-3 6 5 10 5 10V37z" fill="#000"/>
    <path d="M20 8h5" stroke-linejoin="miter"/>
    <path d="M32 29.5s8.5-4 6-9.9C34.1 13 25 13.5 22.5 15h.01C20 13.5 11 13 8.9 19.5c-2.5 5.9 5.9 10 6 10M11.5 30c5.5-3 15.5-3 21 0M11.5 33.5c5.5-3 15.5-3 21 0M11.5 37c5.5-3 15.5-3 21 0" stroke="#fff"/>
  </g>`,
  bQ: `<g fill="#000" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <g stroke="none">
      <circle cx="6" cy="12" r="2.75"/>
      <circle cx="14" cy="9" r="2.75"/>
      <circle cx="22.5" cy="8" r="2.75"/>
      <circle cx="31" cy="9" r="2.75"/>
      <circle cx="39" cy="12" r="2.75"/>
    </g>
    <path d="M9 26c8.5-8.5 15-6.5 23 0M9 26c8.5-13.5 28-13.5 30 0" stroke="#fff" stroke-linecap="butt"/>
    <path d="M11.5 30c6-5 15-5 21 0" stroke="#fff" stroke-linecap="butt"/>
    <path d="M11 38.5c5.5-3.5 15.5-3.5 21 0V36c-6-3-15-3-21 0v2.5z" stroke="none"/>
    <path d="M11 29.5c6-4.5 15-4.5 21 0V26l-3 1-3.5-4.5-4 5-4-5L14 27l-3-1v3.5z" stroke-linecap="butt" stroke-linejoin="miter"/>
    <path d="M11 38.5c0-3.5 3-5 9-5h3c6 0 9 1.5 9 5" stroke="#fff" stroke-linecap="butt"/>
  </g>`,
  bR: `<g fill="#000" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M9 39h27v-3H9v3zM12.5 32l1.5-2.5h17l1.5 2.5h-20zM12 36v-4h21v4H12z" stroke-linecap="butt" stroke-linejoin="miter"/>
    <path d="M14 29.5v-13h17v13H14z" stroke-linecap="butt" stroke-linejoin="miter"/>
    <path d="M14 16.5l-2-2h-1V9h4v2h5V9h5v2h5V9h4v5.5l-2 2H14z" stroke-linecap="butt" stroke-linejoin="miter"/>
    <path d="M11 14h23M9 39h27" fill="none" stroke="#fff" stroke-linejoin="miter"/>
    <path d="M12 36h21M13 32h19M14 29.5h17M14 16.5h17" fill="none" stroke="#fff"/>
  </g>`,
  bB: `<g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <g fill="#000" stroke-linecap="butt">
      <path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2z"/>
      <path d="M15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z"/>
      <path d="M25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z"/>
    </g>
    <path d="M17.5 26h10M15 30h15M22.5 15.5v5M20 18h5" stroke="#fff" stroke-linejoin="miter"/>
  </g>`,
  bN: `<g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21" fill="#000"/>
    <path d="M24 18c.38 5.12-1.5 7-7.5 7.5c4 3 7.2 1 11 5.5c-3 3.5-1 8.5 4 9.5" fill="#000"/>
    <path d="M9.5 25.5a.5.5 0 1 1-1 0 .5.5 0 0 1 1 0z" fill="#fff" stroke="#fff"/>
    <path d="M14.933 15.75a1.75 1.75 0 1 1-3.5 0 1.75 1.75 0 0 1 3.5 0z" fill="#fff" stroke="#fff" stroke-width="1"/>
    <path d="M20.5 12c0 1 1 2 1 2v3l-1.5.5c-1.5 0-3.5.5-5 2s-3 3.5-2.5 5c.5 1.5 2.5.5 2.5.5" fill="#000" stroke="#fff"/>
    <path d="M12.5 20c-1.5 1-4.5 1-4.5 3 0 1 1.5 2 4 2.5" fill="#000" stroke="#fff" stroke-linejoin="round"/>
  </g>`,
  bP: `<g fill="#000" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03C15.41 27.09 11 31.58 11 39.5H34c0-7.92-4.41-12.41-7.41-13.47C28.06 24.84 29 23.03 29 21c0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z"/>
  </g>`,
};

function makePieceSvg(key) {
  const inner = _PATHS[key] || '';
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45">${inner}</svg>`;
  return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
}

// Pre-build all 12 data URIs
const PIECE_URIS = {};
Object.keys(_PATHS).forEach(k => { PIECE_URIS[k] = makePieceSvg(k); });

function pieceTheme(piece) {
  return PIECE_URIS[piece] || '';
}
