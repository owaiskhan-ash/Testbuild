/* REDPOLY interactions plus expressive, reduced-motion-aware movement. */
(function () {
  var root = document.documentElement;
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!reduceMotion) root.classList.add('motion-ready');

  var toast = document.getElementById('toast');
  var toastTimer = null;
  function showToast(msg) {
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toast.classList.remove('show'); }, 2200);
  }

  // Add to cart: quick button confirmation + rupee-price toast.
  document.querySelectorAll('.add-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var name = btn.getAttribute('data-name') || 'FIGURE';
      var original = btn.innerHTML;
      btn.classList.add('added');
      btn.innerHTML = '✓ ADDED TO CART';
      showToast(name + ' added to cart — ₹3,299');
      setTimeout(function () {
        btn.classList.remove('added');
        btn.innerHTML = original;
      }, 1800);
    });
  });

  // Wishlist hearts.
  document.querySelectorAll('.heart').forEach(function (btn) {
    btn.setAttribute('aria-pressed', 'false');
    btn.addEventListener('click', function () {
      var favorite = btn.classList.toggle('is-favorite');
      btn.setAttribute('aria-pressed', favorite ? 'true' : 'false');
      var card = btn.closest('.p-card');
      var name = card && card.querySelector('h3') ? card.querySelector('h3').textContent : 'FIGURE';
      showToast(favorite ? name + ' added to wishlist' : name + ' removed from wishlist');
    });
  });

  // Crew form.
  var form = document.getElementById('crewForm');
  if (form) {
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var email = document.getElementById('crewEmail');
      if (email && email.value) {
        showToast('Welcome to the crew — check your inbox!');
        form.reset();
      }
    });
  }

  // Split only selected text nodes into accessible, staggered words for kinetic type.
  var textEls = Array.prototype.slice.call(document.querySelectorAll(
    '.hero__copy, .limited__title, .limited__sub, .limited__copy, .section-head__title, .section-head__sub, .crew__left h2, .p-card__body h3'
  ));
  if (!reduceMotion) {
    textEls.forEach(function (el) {
      el.classList.add('text-reveal');
      var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
      var nodes = [];
      while (walker.nextNode()) {
        if (walker.currentNode.nodeValue.trim()) nodes.push(walker.currentNode);
      }
      var wordIndex = 0;
      nodes.forEach(function (node) {
        var pieces = node.nodeValue.split(/(\s+)/);
        var fragment = document.createDocumentFragment();
        pieces.forEach(function (piece) {
          if (!piece) return;
          if (/^\s+$/.test(piece)) {
            fragment.appendChild(document.createTextNode(piece));
          } else {
            var word = document.createElement('span');
            word.className = 'motion-word';
            word.style.setProperty('--word-index', wordIndex++);
            word.textContent = piece;
            fragment.appendChild(word);
          }
        });
        node.parentNode.replaceChild(fragment, node);
      });
    });
  }

  // Scroll reveals: cards stagger, copy reveals with its own word rhythm, and the torn edge wipes once.
  var revealEls = Array.prototype.slice.call(document.querySelectorAll(
    '.p-card, .c-card, .limited__center, .limited__right, .section-head, .limited__left, .crew__inner'
  ));
  document.querySelectorAll('.cards .p-card').forEach(function (el, i) {
    el.style.setProperty('--reveal-index', i);
  });
  document.querySelectorAll('.collections__grid .c-card').forEach(function (el, i) {
    el.style.setProperty('--reveal-index', i);
  });
  var limitedSection = document.querySelector('.limited');
  revealEls.forEach(function (el) { el.setAttribute('data-reveal', ''); });
  var targets = revealEls.concat(textEls);
  if (limitedSection) targets.push(limitedSection);
  targets = targets.filter(function (el, i) { return targets.indexOf(el) === i; });

  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        if (entry.target === limitedSection) {
          entry.target.classList.add('is-visible');
        } else if (entry.target.classList.contains('text-reveal')) {
          entry.target.classList.add('text-in');
        } else {
          entry.target.classList.add('in');
        }
        io.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -4% 0px' });
    targets.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add('in'); });
    textEls.forEach(function (el) { el.classList.add('text-in'); });
    if (limitedSection) limitedSection.classList.add('is-visible');
  }

  // Pointer tilt on desktop product cards; intentionally disabled for touch/reduced motion.
  var finePointer = window.matchMedia && window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  if (!reduceMotion && finePointer) {
    document.querySelectorAll('.p-card').forEach(function (card) {
      card.addEventListener('pointermove', function (event) {
        var rect = card.getBoundingClientRect();
        var x = (event.clientX - rect.left) / rect.width - 0.5;
        var y = (event.clientY - rect.top) / rect.height - 0.5;
        card.style.setProperty('--tilt-x', (x * 7).toFixed(2) + 'deg');
        card.style.setProperty('--tilt-y', (y * -6).toFixed(2) + 'deg');
      });
      card.addEventListener('pointerleave', function () {
        card.style.setProperty('--tilt-x', '0deg');
        card.style.setProperty('--tilt-y', '0deg');
      });
    });
  }
})();
