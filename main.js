/* Artisan — cinematic effects: hero parallax + scroll reveals.
   Everything here is progressive enhancement; the page works without it. */
(function () {
  'use strict';

  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var root = document.documentElement;

  /* Remember that the opening curtain has played this session. */
  try { sessionStorage.setItem('artisan-intro', '1'); } catch (e) {}

  /* ---------- Scroll reveals ---------- */
  var items = [].slice.call(document.querySelectorAll('[data-reveal]'));
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

  /* ---------- Hero parallax (scroll + pointer) ---------- */
  var hero = document.querySelector('.hero');
  var layers = [].slice.call(document.querySelectorAll('.layer[data-depth]'));
  if (reduce || !hero || !layers.length) return;

  var finePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  var targetX = 0, targetY = 0, curX = 0, curY = 0, scrollY = 0, ticking = false;

  function frame() {
    curX += (targetX - curX) * 0.07;
    curY += (targetY - curY) * 0.07;
    var unit = hero.offsetWidth / 100;           // 1% of hero width
    var sy = Math.min(scrollY, hero.offsetHeight * 1.2);
    for (var i = 0; i < layers.length; i++) {
      var d = parseFloat(layers[i].getAttribute('data-depth'));
      var x = -curX * d * unit * 2.6;
      var y = -curY * d * unit * 1.2 + sy * (1 - d) * 0.18;   // far layers lag behind the scroll
      layers[i].style.transform = 'translate3d(' + x.toFixed(2) + 'px,' + y.toFixed(2) + 'px,0)';
    }
    if (Math.abs(targetX - curX) > 0.001 || Math.abs(targetY - curY) > 0.001) {
      requestAnimationFrame(frame);
    } else {
      ticking = false;
    }
  }
  function kick() { if (!ticking) { ticking = true; requestAnimationFrame(frame); } }

  window.addEventListener('scroll', function () {
    scrollY = window.pageYOffset || 0;
    kick();
  }, { passive: true });

  if (finePointer) {
    hero.addEventListener('pointermove', function (e) {
      var r = hero.getBoundingClientRect();
      targetX = (e.clientX - r.left) / r.width - 0.5;
      targetY = (e.clientY - r.top) / r.height - 0.5;
      kick();
    });
    hero.addEventListener('pointerleave', function () { targetX = 0; targetY = 0; kick(); });
  }

  scrollY = window.pageYOffset || 0;
  kick();
  root.classList.add('fx-ready');
})();
