'use strict';
console.log('puzzle.js VERSION 4 loaded');

// ── Config ─────────────────────────────────────────────────────────────────
const API = 'http://localhost:5000';

// Unicode chess pieces
const GLYPHS = {
  wK:'♔', wQ:'♕', wR:'♖', wB:'♗', wN:'♘', wP:'♙',
  bK:'♚', bQ:'♛', bR:'♜', bB:'♝', bN:'♞', bP:'♟',
};

// ── Demo puzzles (fallback when server not running) ────────────────────────
const DEMO_PUZZLES = [
  { id:'000Pw', fen:'6k1/5p1p/4p3/4q3/3nN3/2Q3P1/PP3P1P/6K1 w - - 2 37',
    moves:['e4d2','d4e2','g1f1','e2c3'], rating:1550,
    ratingDeviation:75, popularity:92, nbPlays:626,
    themes:['fork','endgame'], primaryCategory:'Fork',
    difficultyTier:'Advanced', categories:['Fork','Endgame'],
    gameUrl:'https://lichess.org/au2lCK5o#73' },
  { id:'0000D', fen:'5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 1 26',
    moves:['d3d6','f8d8','d6d8','f6d8'], rating:1579,
    ratingDeviation:73, popularity:96, nbPlays:36672,
    themes:['endgame'], primaryCategory:'Endgame',
    difficultyTier:'Advanced', categories:['Endgame'],
    gameUrl:'https://lichess.org/F8M8OS71#53' },
  { id:'003xY', fen:'6k1/pp3p1p/2p3p1/2b5/2Bn4/1P4P1/P4P1P/3R2K1 b - - 0 25',
    moves:['d4f3','g1f1','c5e3','d1d8'], rating:1450,
    ratingDeviation:78, popularity:85, nbPlays:3400,
    themes:['fork','endgame'], primaryCategory:'Fork',
    difficultyTier:'Intermediate', categories:['Fork','Endgame'],
    gameUrl:'' },
  { id:'004rZ', fen:'r1b1kb1r/ppqn1ppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R w KQkq - 2 7',
    moves:['d4d5','c6d5','c4d5','e6d5','c3d5','f6d5','d1d5'],
    rating:1720, ratingDeviation:77, popularity:88, nbPlays:4100,
    themes:['discoveredAttack','middlegame'], primaryCategory:'Discovered Attack',
    difficultyTier:'Hard', categories:['Discovered Attack'],
    gameUrl:'' },
  { id:'005pN', fen:'r4rk1/1b2ppbp/pq1p1np1/1p6/3BP3/1BN2P2/PPP1Q1PP/R4RK1 b - - 0 15',
    moves:['b6e3','e2e3','f6e4','c3e4','g7b2'],
    rating:1830, ratingDeviation:80, popularity:91, nbPlays:6200,
    themes:['pin','sacrifice','middlegame'], primaryCategory:'Pin',
    difficultyTier:'Hard', categories:['Pin','Sacrifice'],
    gameUrl:'' },
  { id:'006mQ', fen:'r1bqr1k1/ppp2ppp/2n5/3p4/1bBP4/2N1PN2/PP3PPP/R1BQK2R w KQ - 1 10',
    moves:['c3d5','c6d4','d5f4','d4c2'],
    rating:1380, ratingDeviation:73, popularity:90, nbPlays:8900,
    themes:['fork','middlegame'], primaryCategory:'Fork',
    difficultyTier:'Intermediate', categories:['Fork'],
    gameUrl:'' },
  { id:'007hK', fen:'8/8/8/8/3k4/8/3KR3/8 w - - 0 1',
    moves:['e2e4','d4d3','e4e3'],
    rating:950, ratingDeviation:80, popularity:85, nbPlays:2100,
    themes:['rookEndgame','endgame'], primaryCategory:'Rook Endgame',
    difficultyTier:'Easy', categories:['Endgame','Rook Endgame'],
    gameUrl:'' },
  { id:'008aB', fen:'r2qkb1r/pp3ppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP3PPP/RN1QK2R b KQkq - 0 8',
    moves:['c5d4','c3d4','f6e4','d4e5','d8a5'],
    rating:1620, ratingDeviation:76, popularity:89, nbPlays:5300,
    themes:['hangingPiece','middlegame'], primaryCategory:'Hanging Piece',
    difficultyTier:'Advanced', categories:['Hanging Piece'],
    gameUrl:'' },
];

// ── State ──────────────────────────────────────────────────────────────────
let chess       = null;
let puzzle      = null;
let moveIndex   = 0;
let playerColor = 'w';
let orientation = 'white'; // 'white' | 'black'
let hintUsed    = false;
let selected    = null;    // currently selected square string e.g. 'e4'
let highlights  = {};      // square → css class

// ── DOM ────────────────────────────────────────────────────────────────────
const $boardEl       = document.getElementById('board');
const $feedbackBanner= document.getElementById('feedbackBanner');
const $puzzleId      = document.getElementById('puzzleId');
const $gameLink      = document.getElementById('gameLink');
const $diffBadge     = document.getElementById('difficultyBadge');
const $ratingValue   = document.getElementById('ratingValue');
const $playsLabel    = document.getElementById('playsLabel');
const $themesRow     = document.getElementById('themesRow');
const $turnDot       = document.getElementById('turnDot');
const $turnText      = document.getElementById('turnText');
const $progressBar   = document.getElementById('progressBar');
const $progressFrac  = document.getElementById('progressFraction');
const $moveDots      = document.getElementById('moveDots');
const $moveLog       = document.getElementById('moveLog');
const $solvedCard    = document.getElementById('solvedCard');
const $solvedSub     = document.getElementById('solvedSub');
const $loadingOverlay= document.getElementById('loadingOverlay');
const $boardWrap     = document.querySelector('.board-wrap');

// ── Board rendering ────────────────────────────────────────────────────────

function buildBoard() {
  $boardEl.innerHTML = '';
  $boardEl.className = 'chess-board';

  const files = ['a','b','c','d','e','f','g','h'];
  const ranks = orientation === 'white' ? [8,7,6,5,4,3,2,1] : [1,2,3,4,5,6,7,8];
  const filesOrd = orientation === 'white' ? files : [...files].reverse();

  ranks.forEach(rank => {
    filesOrd.forEach(file => {
      const sq = file + rank;
      const light = (files.indexOf(file) + rank) % 2 !== 0;
      const cell = document.createElement('div');
      cell.className = 'sq ' + (light ? 'sq-light' : 'sq-dark');
      cell.dataset.sq = sq;

      // Rank label (leftmost column)
      if ((orientation === 'white' && file === 'a') ||
          (orientation === 'black' && file === 'h')) {
        const lbl = document.createElement('span');
        lbl.className = 'coord coord-rank';
        lbl.textContent = rank;
        cell.appendChild(lbl);
      }
      // File label (bottom row)
      if ((orientation === 'white' && rank === 1) ||
          (orientation === 'black' && rank === 8)) {
        const lbl = document.createElement('span');
        lbl.className = 'coord coord-file';
        lbl.textContent = file;
        cell.appendChild(lbl);
      }

      cell.addEventListener('click', onSquareClick);
      cell.addEventListener('dragover', e => e.preventDefault());
      cell.addEventListener('drop', onDrop);
      $boardEl.appendChild(cell);
    });
  });
}

function renderPieces() {
  document.querySelectorAll('.sq').forEach(cell => {
    // Remove old piece
    cell.querySelectorAll('.piece').forEach(p => p.remove());
    // Apply highlight classes
    const sq = cell.dataset.sq;
    cell.classList.remove('hl-from','hl-to','hl-hint','hl-legal');
    if (highlights[sq]) cell.classList.add(highlights[sq]);

    const piece = chess.get(sq);
    if (!piece) return;

    const key = piece.color + piece.type.toUpperCase();
    const el = document.createElement('span');
    el.className = 'piece piece-' + piece.color;
    el.textContent = GLYPHS[key] || '?';
    el.draggable = true;
    el.dataset.sq = sq;
    el.addEventListener('dragstart', onDragStart);
    cell.appendChild(el);
  });
}

function renderBoard() {
  renderPieces();
}

// ── Highlight helpers ──────────────────────────────────────────────────────

function clearHighlights() {
  highlights = {};
  selected = null;
}

function setHL(sq, cls) { highlights[sq] = cls; }

// ── Drag & drop ────────────────────────────────────────────────────────────

let dragFromSq = null;

function onDragStart(e) {
  const sq = e.target.dataset.sq;
  const piece = chess.get(sq);
  if (!piece) return;
  if (moveIndex >= puzzle.moves.length || moveIndex % 2 === 0) { e.preventDefault(); return; }
  if (piece.color !== playerColor) { e.preventDefault(); return; }
  dragFromSq = sq;
  e.dataTransfer.effectAllowed = 'move';
  e.dataTransfer.setData('text/plain', sq);
}

function onDrop(e) {
  e.preventDefault();
  const toSq = e.currentTarget.dataset.sq;
  if (!dragFromSq || !toSq) return;
  tryMove(dragFromSq, toSq);
  dragFromSq = null;
}

// ── Click to move ──────────────────────────────────────────────────────────

function onSquareClick(e) {
  const sq = e.currentTarget.dataset.sq;
  if (!puzzle || moveIndex >= puzzle.moves.length || moveIndex % 2 === 0) return;

  if (selected) {
    if (sq === selected) {
      clearHighlights();
      renderBoard();
      return;
    }
    // Clicking a new own piece — reselect
    const clickedPiece = chess.get(sq);
    if (clickedPiece && clickedPiece.color === playerColor) {
      selectSquare(sq);
      return;
    }
    // Try the move
    tryMove(selected, sq);
  } else {
    const piece = chess.get(sq);
    if (!piece || piece.color !== playerColor) return;
    selectSquare(sq);
  }
}

function selectSquare(sq) {
  clearHighlights();
  selected = sq;
  setHL(sq, 'hl-from');
  // Show legal move targets
  chess.moves({ square: sq, verbose: true }).forEach(m => setHL(m.to, 'hl-legal'));
  renderBoard();
}

// ── Move logic ─────────────────────────────────────────────────────────────

function tryMove(from, to) {
  if (moveIndex % 2 === 0 || moveIndex >= puzzle.moves.length) return;

  const expected = puzzle.moves[moveIndex];
  const expFrom  = expected.slice(0,2);
  const expTo    = expected.slice(2,4);

  // Detect promotion
  const piece = chess.get(from);
  const isPromo = piece && piece.type === 'p' && (to[1]==='8'||to[1]==='1');
  const promo = expected.length > 4 ? expected[4] : 'q';

  const result = chess.move({ from, to, promotion: promo });
  if (!result) {
    clearHighlights();
    renderBoard();
    return;
  }

  const correct = from === expFrom && to === expTo;

  clearHighlights();
  setHL(from, 'hl-from');
  setHL(to,   'hl-to');

  if (correct) {
    appendLog(result.san, false, true);
    updateMoveDots(moveIndex, 'correct');
    moveIndex++;
    updateProgress();
    renderBoard();

    if (moveIndex >= puzzle.moves.length) {
      finishPuzzle(true);
    } else {
      showFeedback('✓ Correct!', 'correct');
      flashBoard('correct');
      updateTurnIndicator();
      playAutoMove(650);
    }
  } else {
    chess.undo();
    appendLog(result.san, false, false);
    updateMoveDots(moveIndex, 'wrong');
    showFeedback('✗ Not the best move — try again', 'wrong');
    flashBoard('wrong');
    clearHighlights();
    renderBoard();
  }
}

// ── Auto-play (opponent move) ──────────────────────────────────────────────

function playAutoMove(delay) {
  delay = (typeof delay === 'number') ? delay : 400;
  console.log('[AUTO] scheduling in', delay, 'ms, moveIndex=', moveIndex);
  window.setTimeout(function() {
    console.log('[AUTO] fired, moveIndex=', moveIndex, 'puzzle=', puzzle && puzzle.id);
    if (!puzzle || moveIndex >= puzzle.moves.length) return;
    var uci    = puzzle.moves[moveIndex];
    var from   = uci.slice(0, 2);
    var to     = uci.slice(2, 4);
    var promo  = uci.length > 4 ? uci[4] : 'q';
    console.log('[AUTO] moving', from, '->', to, 'chess.turn()=', chess.turn());
    var result = chess.move({ from: from, to: to, promotion: promo });
    console.log('[AUTO] result=', result ? result.san : 'NULL');
    if (!result) return;

    clearHighlights();
    setHL(from, 'hl-from');
    setHL(to,   'hl-to');
    renderBoard();
    appendLog(result.san, true, null);
    moveIndex++;
    updateProgress();
    updateMoveDots(moveIndex - 1, 'auto');
    updateTurnIndicator();
  }, delay);
}

// ── UI updates ─────────────────────────────────────────────────────────────

function showLoading(on) {
  $loadingOverlay.classList.toggle('hidden', !on);
}

function flashBoard(type) {
  $boardWrap.classList.remove('correct','wrong');
  void $boardWrap.offsetWidth;
  $boardWrap.classList.add(type);
  setTimeout(() => $boardWrap.classList.remove(type), 900);
}

function showFeedback(text, type) {
  $feedbackBanner.textContent = text;
  $feedbackBanner.className = `feedback-banner show-${type}`;
  setTimeout(() => { $feedbackBanner.className = 'feedback-banner'; }, 2400);
}

function updateTurnIndicator() {
  if (!puzzle || moveIndex >= puzzle.moves.length) return;
  const turn = chess.turn();
  $turnDot.className = `turn-dot ${turn === 'w' ? 'white' : 'black'}`;
  if (turn === playerColor) {
    $turnText.textContent = `${turn === 'w' ? 'White' : 'Black'} to move — find the best move`;
  } else {
    $turnText.textContent = 'Opponent is thinking…';
  }
}

function buildMoveDots(moves) {
  $moveDots.innerHTML = '';
  moves.forEach((_, i) => {
    const dot = document.createElement('div');
    dot.className = 'move-dot';
    dot.dataset.index = i;
    $moveDots.appendChild(dot);
  });
}

function updateMoveDots(index, status) {
  const dot = $moveDots.querySelector(`[data-index="${index}"]`);
  if (!dot) return;
  dot.classList.remove('auto','pending','correct','wrong');
  dot.classList.add(status);
}

function updateProgress() {
  if (!puzzle) return;
  const total = puzzle.moves.length;
  const playerTotal = Math.ceil((total - 1) / 2);
  const playerDone  = Math.floor(moveIndex / 2);
  const pct = total <= 1 ? 100 : Math.min(100, (moveIndex / (total - 1)) * 100);
  $progressBar.style.width = pct + '%';
  $progressFrac.textContent = `${playerDone} / ${playerTotal}`;
}

function appendLog(san, isAuto, isCorrect) {
  const entry = document.createElement('div');
  const num   = document.createElement('span');
  const move  = document.createElement('span');
  const tag   = document.createElement('span');
  num.className   = 'move-num';
  num.textContent = `${$moveLog.children.length + 1}.`;
  move.className  = 'move-san';
  move.textContent = san;
  tag.className   = 'move-tag';
  if (isAuto)         { entry.className = 'log-entry auto-move';      tag.textContent = 'auto'; }
  else if (isCorrect) { entry.className = 'log-entry player-correct'; tag.textContent = '✓'; }
  else                { entry.className = 'log-entry player-wrong';   tag.textContent = '✗'; }
  entry.append(num, move, tag);
  $moveLog.appendChild(entry);
  $moveLog.scrollTop = $moveLog.scrollHeight;
}

function populateInfo(p) {
  $puzzleId.textContent = p.id;
  $gameLink.href  = p.gameUrl || '#';
  $gameLink.style.display = p.gameUrl ? '' : 'none';

  const tier = (p.difficultyTier || 'intermediate').toLowerCase().replace(' ','-');
  $diffBadge.textContent = p.difficultyTier || '—';
  $diffBadge.className   = `difficulty-badge ${tier}`;
  $ratingValue.textContent = p.rating;
  $playsLabel.textContent  = p.nbPlays ? `${p.nbPlays.toLocaleString()} plays` : '';

  $themesRow.innerHTML = '';
  (p.themes || []).slice(0,8).forEach(t => {
    const tag = document.createElement('span');
    tag.className   = 'theme-tag';
    tag.textContent = camelToWords(t);
    $themesRow.appendChild(tag);
  });
}

function camelToWords(s) { return s.replace(/([A-Z])/g, ' $1').trim(); }

function finishPuzzle(success) {
  flashBoard(success ? 'correct' : 'wrong');
  $turnDot.className = `turn-dot ${success ? 'white' : 'black'}`;
  $turnText.textContent = success ? 'Puzzle solved! 🎉' : 'Puzzle failed.';
  if (success) {
    $solvedSub.textContent = `Rating ${puzzle.rating} · ${puzzle.primaryCategory}${hintUsed ? ' (hint used)' : ''}`;
    $solvedCard.classList.remove('hidden');
  }
}

// ── Rating filter ──────────────────────────────────────────────────────────

function getRatingFilter() {
  const val = document.getElementById('ratingFilter').value;
  if (val === 'all') return { ratingMin: 600, ratingMax: 2400 };
  const [lo, hi] = val.split('-').map(Number);
  return { ratingMin: lo, ratingMax: hi };
}

function demoFallback(ratingMin, ratingMax) {
  const pool = DEMO_PUZZLES.filter(p => p.rating >= ratingMin && p.rating <= ratingMax);
  const src  = pool.length ? pool : DEMO_PUZZLES;
  return src[Math.floor(Math.random() * src.length)];
}

// ── Load puzzle ────────────────────────────────────────────────────────────

async function loadPuzzle() {
  showLoading(true);
  $solvedCard.classList.add('hidden');
  $moveLog.innerHTML = '';
  clearHighlights();

  const { ratingMin, ratingMax } = getRatingFilter();

  try {
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), 2500);
    const res = await fetch(`${API}/api/puzzle/random?ratingMin=${ratingMin}&ratingMax=${ratingMax}`,
      { signal: ctrl.signal });
    if (!res.ok) throw new Error();
    puzzle = await res.json();
  } catch (_) {
    puzzle = demoFallback(ratingMin, ratingMax);
  }

  chess    = new Chess(puzzle.fen);
  moveIndex = 0;
  hintUsed  = false;

  // Player is the side to move AFTER moves[0] (the setup move)
  const startTurn = chess.turn(); // who plays moves[0]
  playerColor = startTurn === 'w' ? 'b' : 'w';
  orientation = playerColor === 'w' ? 'white' : 'black';

  buildBoard();
  renderBoard();
  populateInfo(puzzle);
  buildMoveDots(puzzle.moves);
  updateProgress();

  // Mark move[0] dot immediately as auto
  const d0 = $moveDots.querySelector('[data-index="0"]');
  if (d0) { d0.classList.remove('pending'); d0.classList.add('pending'); }

  $turnDot.className = `turn-dot ${startTurn === 'w' ? 'white' : 'black'}`;
  $turnText.textContent = `${startTurn === 'w' ? 'White' : 'Black'} plays first…`;

  showLoading(false);
  console.log('Puzzle loaded:', puzzle.id, '| moves:', puzzle.moves, '| playerColor:', playerColor);
  playAutoMove(400);
}

// ── Hint ───────────────────────────────────────────────────────────────────

function showHint() {
  if (!puzzle || moveIndex >= puzzle.moves.length || moveIndex % 2 === 0) return;
  const uci = puzzle.moves[moveIndex];
  clearHighlights();
  setHL(uci.slice(0,2), 'hl-hint');
  renderBoard();
  hintUsed = true;
  showFeedback('💡 Move the highlighted piece', 'correct');
}

// ── Event wiring ───────────────────────────────────────────────────────────

document.getElementById('btnNext').addEventListener('click', loadPuzzle);
document.getElementById('btnNextBottom').addEventListener('click', loadPuzzle);
document.getElementById('btnSolvedNext').addEventListener('click', loadPuzzle);
document.getElementById('btnHint').addEventListener('click', showHint);
document.getElementById('btnFlip').addEventListener('click', () => {
  orientation = orientation === 'white' ? 'black' : 'white';
  buildBoard();
  renderBoard();
});
document.getElementById('ratingFilter').addEventListener('change', loadPuzzle);

loadPuzzle();
