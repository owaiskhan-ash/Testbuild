/* Artisan — cinematic effects.
   Everything here is progressive enhancement; the page works without it. */
(function () {
  'use strict';

  var root = document.documentElement;
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var finePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  var $ = function (s, el) { return (el || document).querySelector(s); };
  var $$ = function (s, el) { return [].slice.call((el || document).querySelectorAll(s)); };
  var clamp = function (v, a, b) { return Math.min(b, Math.max(a, v)); };

  /* Remember that the opening curtain has played this session. */
  try { sessionStorage.setItem('artisan-intro', '1'); } catch (e) {}

  var hero = $('.hero');
  var nav = $('.nav');
  var pin = $('.pin');
  var crafts = $('.crafts');
  var cards = $$('.card');
  var layers = $$('.layer[data-depth]');
  var parItems = $$('[data-par]');
  var duo = $('.duo');

  /* ---------- Pinned card sequence: decide before reveals are wired ---------- */
  var pinOn = false;
  function fitsPin() {
    return !reduce && pin && crafts && window.innerWidth > 760 &&
      window.innerHeight >= crafts.offsetHeight + (nav ? nav.offsetHeight : 0) - 2;
  }
  if (fitsPin()) {
    pinOn = true;
    root.classList.add('pin-on');
    cards.forEach(function (c) { c.removeAttribute('data-reveal'); });
  }

  /* ---------- Scroll reveals ---------- */
  var items = $$('[data-reveal]');
  if (reduce || !('IntersectionObserver' in window)) {
    items.forEach(function (el) { el.classList.add('is-in'); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-in');
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.18, rootMargin: '0px 0px -6% 0px' });
    items.forEach(function (el) { io.observe(el); });
  }

  /* ---------- Ambient sound (synthesised, off by default) ---------- */
  setupSound();

  if (reduce) return;

  /* ---------- One scroll/pointer loop drives parallax, dusk, wipes, pin ---------- */
  var targetX = 0, targetY = 0, curX = 0, curY = 0, ticking = false;

  function update() {
    var vh = window.innerHeight;
    var sy = window.pageYOffset || 0;

    if (hero) {
      var hh = hero.offsetHeight;
      root.classList.toggle('hero-off', sy > hh * 1.3);
      /* hero parallax */
      if (sy < hh * 1.3) {
        var unit = hero.offsetWidth / 100;
        for (var i = 0; i < layers.length; i++) {
          var d = parseFloat(layers[i].getAttribute('data-depth'));
          var x = -curX * d * unit * 2.6;
          var y = -curY * d * unit * 1.2 + sy * (1 - d) * 0.18;   // far layers lag behind the scroll
          layers[i].style.transform = 'translate3d(' + x.toFixed(2) + 'px,' + y.toFixed(2) + 'px,0)';
        }
        /* day to dusk as the hero scrolls away */
        hero.style.setProperty('--dusk', clamp((sy - hh * 0.04) / (hh * 0.5), 0, 1).toFixed(3));
      }
    }

    if (crafts) {
      var r = crafts.getBoundingClientRect();
      /* foam retreats from the top edge, washes up at the bottom edge */
      var pt = clamp(r.top / vh, 0, 1);
      var pb = clamp((r.bottom - vh * 0.35) / (vh * 0.65), 0, 1);
      crafts.style.setProperty('--wipe-top', (0.1 + 0.62 * Math.pow(pt, 1.4)).toFixed(3));
      crafts.style.setProperty('--wipe-bottom', (0.1 + 0.48 * (1 - pb)).toFixed(3));

      /* pinned sequence: cards slide in one at a time */
      if (pinOn) {
        var pr = pin.getBoundingClientRect();
        var navH = nav ? nav.offsetHeight : 0;
        var travel = pin.offsetHeight - crafts.offsetHeight;
        // start a little before the section pins so the first card is arriving as it locks
        var p = clamp((navH + vh * 0.28 - pr.top) / (travel + vh * 0.28), 0, 1);
        for (var c = 0; c < cards.length; c++) {
          var e = clamp(p * 3.5 - c * 1.05, 0, 1);
          e = 1 - Math.pow(1 - e, 3);
          cards[c].style.setProperty('--e', e.toFixed(3));
        }
      }
    }

    /* artisans: blobs and figures drift at different speeds */
    if (duo) {
      var dr = duo.getBoundingClientRect();
      if (dr.bottom > -100 && dr.top < vh + 100) {
        var off = (dr.top + dr.height / 2 - vh / 2) / vh;      // -0.5 … 0.5 across the viewport
        var u = duo.offsetWidth / 176;
        for (var k = 0; k < parItems.length; k++) {
          var amt = parseFloat(parItems[k].getAttribute('data-par'));
          parItems[k].style.setProperty('--par', (off * amt * u * 2.2).toFixed(2) + 'px');
        }
      }
    }
  }

  function frame() {
    curX += (targetX - curX) * 0.07;
    curY += (targetY - curY) * 0.07;
    update();
    if (Math.abs(targetX - curX) > 0.001 || Math.abs(targetY - curY) > 0.001) {
      requestAnimationFrame(frame);
    } else {
      ticking = false;
    }
  }
  function kick() { if (!ticking) { ticking = true; requestAnimationFrame(frame); } }

  window.addEventListener('scroll', kick, { passive: true });
  window.addEventListener('resize', function () {
    var want = fitsPin();
    if (pinOn && !want) {              // viewport too small now: release the pin, show the cards
      pinOn = false;
      root.classList.remove('pin-on');
    } else if (!pinOn && want && cards.length && !cards[0].hasAttribute('data-reveal')) {
      pinOn = true;
      root.classList.add('pin-on');
    }
    kick();
  });

  if (finePointer && hero) {
    hero.addEventListener('pointermove', function (e) {
      var r = hero.getBoundingClientRect();
      targetX = (e.clientX - r.left) / r.width - 0.5;
      targetY = (e.clientY - r.top) / r.height - 0.5;
      kick();
    });
    hero.addEventListener('pointerleave', function () { targetX = 0; targetY = 0; kick(); });
  }
  kick();

  if (!finePointer) return;

  /* ---------- Card tilt + light sweep ---------- */
  cards.forEach(function (card) {
    card.addEventListener('pointermove', function (e) {
      var r = card.getBoundingClientRect();
      var px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
      card.style.setProperty('--ry', ((px - 0.5) * 11).toFixed(2) + 'deg');
      card.style.setProperty('--rx', ((0.5 - py) * 9).toFixed(2) + 'deg');
      card.style.setProperty('--sx', (px * 100).toFixed(1) + '%');
      card.style.setProperty('--sy', (py * 100).toFixed(1) + '%');
      card.style.setProperty('--ix', ((0.5 - px) * 3.2).toFixed(2) + '%');
      card.style.setProperty('--iy', ((0.5 - py) * 3.2).toFixed(2) + '%');
    });
    card.addEventListener('pointerleave', function () {
      ['--rx', '--ry', '--ix', '--iy'].forEach(function (v) { card.style.removeProperty(v); });
    });
  });

  /* ---------- Magnetic buttons ---------- */
  var magnets = $$('.btn, .dot').map(function (el) { return { el: el, on: false }; });
  window.addEventListener('pointermove', function (e) {
    for (var i = 0; i < magnets.length; i++) {
      var m = magnets[i], r = m.el.getBoundingClientRect();
      // measure from the resting position, not the pulled one
      var ox = parseFloat(m.el.style.getPropertyValue('--mx')) || 0;
      var oy = parseFloat(m.el.style.getPropertyValue('--my')) || 0;
      var cx = r.left + r.width / 2 - ox, cy = r.top + r.height / 2 - oy;
      var dx = e.clientX - cx, dy = e.clientY - cy;
      var reach = Math.max(r.width, r.height) * 0.5 + 46;
      if (Math.abs(dx) < reach && Math.abs(dy) < reach) {
        m.on = true;
        m.el.style.setProperty('--mx', (dx * 0.32).toFixed(1) + 'px');
        m.el.style.setProperty('--my', (dy * 0.32).toFixed(1) + 'px');
      } else if (m.on) {
        m.on = false;
        m.el.style.removeProperty('--mx');
        m.el.style.removeProperty('--my');
      }
    }
  }, { passive: true });

  /* ---------- Custom cursor ---------- */
  var dot = document.createElement('div'); dot.className = 'cursor-dot';
  var ring = document.createElement('div'); ring.className = 'cursor-ring';
  document.body.appendChild(dot); document.body.appendChild(ring);
  root.classList.add('has-cursor');
  var mx = 0, my = 0, rx = 0, ry = 0, cursorLoop = false;
  function cursorFrame() {
    rx += (mx - rx) * 0.2; ry += (my - ry) * 0.2;
    ring.style.transform = 'translate3d(' + rx.toFixed(1) + 'px,' + ry.toFixed(1) + 'px,0)';
    if (Math.abs(mx - rx) > 0.2 || Math.abs(my - ry) > 0.2) requestAnimationFrame(cursorFrame);
    else cursorLoop = false;
  }
  window.addEventListener('pointermove', function (e) {
    if (e.pointerType && e.pointerType !== 'mouse') return;
    mx = e.clientX; my = e.clientY;
    if (!root.classList.contains('cursor-on')) { rx = mx; ry = my; root.classList.add('cursor-on'); }
    dot.style.transform = 'translate3d(' + mx + 'px,' + my + 'px,0)';
    var t = e.target && e.target.closest ? e.target : null;
    root.classList.toggle('cursor-hover', !!(t && t.closest('a, button')));
    root.classList.toggle('cursor-text', !!(t && t.closest('input, select, textarea')));
    if (!cursorLoop) { cursorLoop = true; requestAnimationFrame(cursorFrame); }
  }, { passive: true });
  document.addEventListener('pointerleave', function () { root.classList.remove('cursor-on'); });
  document.addEventListener('mouseleave', function () { root.classList.remove('cursor-on'); });

  /* ---------- Ambient sound: lake lapping + breeze, generated with Web Audio ---------- */
  function setupSound() {
    var btn = $('.sound');
    var AC = window.AudioContext || window.webkitAudioContext;
    if (!btn) return;
    if (!AC) { btn.style.display = 'none'; return; }
    var ctx = null, master = null, playing = false;

    function noiseBuffer(seconds, brown) {
      var len = Math.floor(ctx.sampleRate * seconds);
      var buf = ctx.createBuffer(1, len, ctx.sampleRate), data = buf.getChannelData(0), last = 0;
      for (var i = 0; i < len; i++) {
        var white = Math.random() * 2 - 1;
        if (brown) { last = (last + 0.02 * white) / 1.02; data[i] = last * 3.5; }
        else { data[i] = white; }
      }
      return buf;
    }
    function lfo(freq, depth, target) {
      var o = ctx.createOscillator(), g = ctx.createGain();
      o.frequency.value = freq; g.gain.value = depth;
      o.connect(g); g.connect(target); o.start();
    }
    function build() {
      ctx = new AC();
      master = ctx.createGain(); master.gain.value = 0; master.connect(ctx.destination);

      /* water: low brown noise, swelling slowly like small waves on the shore */
      var water = ctx.createBufferSource(); water.buffer = noiseBuffer(6, true); water.loop = true;
      var wf = ctx.createBiquadFilter(); wf.type = 'lowpass'; wf.frequency.value = 520;
      var wg = ctx.createGain(); wg.gain.value = 0.5;
      water.connect(wf); wf.connect(wg); wg.connect(master);
      lfo(0.17, 0.28, wg.gain); lfo(0.07, 180, wf.frequency);
      water.start();

      /* breeze: airy band of white noise drifting in pitch and level */
      var wind = ctx.createBufferSource(); wind.buffer = noiseBuffer(5, false); wind.loop = true;
      var bf = ctx.createBiquadFilter(); bf.type = 'bandpass'; bf.frequency.value = 900; bf.Q.value = 0.6;
      var bg = ctx.createGain(); bg.gain.value = 0.05;
      wind.connect(bf); bf.connect(bg); bg.connect(master);
      lfo(0.05, 0.035, bg.gain); lfo(0.031, 350, bf.frequency);
      wind.start();
    }
    btn.addEventListener('click', function () {
      if (!ctx) build();
      playing = !playing;
      if (playing && ctx.state === 'suspended') ctx.resume();
      var now = ctx.currentTime;
      master.gain.cancelScheduledValues(now);
      master.gain.setValueAtTime(master.gain.value, now);
      master.gain.linearRampToValueAtTime(playing ? 0.55 : 0, now + (playing ? 1.8 : 0.7));
      if (!playing) setTimeout(function () { if (!playing && ctx.state === 'running') ctx.suspend(); }, 800);
      btn.setAttribute('aria-pressed', String(playing));
      btn.setAttribute('aria-label', playing ? 'Turn ambient sound off' : 'Turn ambient sound on');
    });
  }
})();
