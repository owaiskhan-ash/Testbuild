/* REDPOLY interactions plus expressive, reduced-motion-aware movement. */
(function () {
  var root = document.documentElement;
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!reduceMotion) root.classList.add('motion-ready');

  // Reveal each character-page title glyph in sequence while keeping one clean accessible label.
  if (!reduceMotion) {
    document.querySelectorAll('.char-hero h1[aria-label]').forEach(function (heading) {
      var walker = document.createTreeWalker(heading, NodeFilter.SHOW_TEXT);
      var textNodes = [];
      while (walker.nextNode()) textNodes.push(walker.currentNode);
      var letterIndex = 0;
      textNodes.forEach(function (node) {
        var fragment = document.createDocumentFragment();
        Array.from(node.nodeValue).forEach(function (character) {
          if (/\s/.test(character)) {
            fragment.appendChild(document.createTextNode(character));
            return;
          }
          var letter = document.createElement('span');
          letter.className = 'char-title__letter';
          letter.setAttribute('aria-hidden', 'true');
          letter.style.setProperty('--letter-index', letterIndex++);
          letter.textContent = character;
          fragment.appendChild(letter);
        });
        node.parentNode.replaceChild(fragment, node);
      });
    });
  }

  // Warm the adjacent page and its large hero art on intent so the color wipe never lands on an unloaded screen.
  var productPagerLinks = document.querySelectorAll('.char-product-pager a[data-transition-color]');
  var supportsColorWipe = window.CSS && window.CSS.supports && window.CSS.supports('clip-path', 'circle(1px at 50% 50%)');
  if (productPagerLinks.length) {
    var pagePreloads = Object.create(null);
    function prepareProductPage(link) {
      var href = link.href;
      if (pagePreloads[href]) return pagePreloads[href];
      var pageHint = document.createElement('link');
      pageHint.rel = 'prefetch';
      pageHint.href = href;
      document.head.appendChild(pageHint);
      pagePreloads[href] = new Promise(function (resolve) {
        var image = new Image();
        image.decoding = 'async';
        if ('fetchPriority' in image) image.fetchPriority = 'high';
        function imageReady() {
          if (typeof image.decode === 'function') image.decode().then(resolve, resolve);
          else resolve();
        }
        image.onload = imageReady;
        image.onerror = resolve;
        image.src = link.getAttribute('data-transition-image');
        if (image.complete && image.naturalWidth) imageReady();
      });
      return pagePreloads[href];
    }

    productPagerLinks.forEach(function (link) {
      link.addEventListener('pointerenter', function () { prepareProductPage(link); }, { once: true });
      link.addEventListener('focus', function () { prepareProductPage(link); }, { once: true });
      link.addEventListener('touchstart', function () { prepareProductPage(link); }, { once: true, passive: true });
    });

    if (!reduceMotion && supportsColorWipe) {
      var pageWipeStarted = false;
      productPagerLinks.forEach(function (link) {
        link.addEventListener('click', function (event) {
          if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || link.target === '_blank' || link.hasAttribute('download')) return;
          var destination = new URL(link.href, window.location.href);
          if (destination.origin !== window.location.origin || pageWipeStarted) return;
          event.preventDefault();
          pageWipeStarted = true;
          var wipe = document.createElement('div');
          wipe.className = 'char-page-color-wipe';
          wipe.setAttribute('aria-hidden', 'true');
          wipe.style.setProperty('--wipe-color', link.getAttribute('data-transition-color'));
          wipe.style.setProperty('--wipe-origin', link.getAttribute('data-transition-direction') === 'previous' ? '0% 50%' : '100% 50%');
          document.body.appendChild(wipe);
          var wipeFinished = new Promise(function (resolve) {
            var fallback = window.setTimeout(resolve, 650);
            wipe.addEventListener('animationend', function () {
              window.clearTimeout(fallback);
              resolve();
            }, { once: true });
          });
          prepareProductPage(link);
          wipeFinished.then(function () {
            window.location.assign(destination.href);
          });
        });
      });
    }
  }

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
      showToast(name + ' added to cart - ₹3,299');
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
        showToast('Welcome to the crew - check your inbox!');
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

  // Character product pages: staged scroll reveals and a subtle 3D response to desktop pointer movement.
  var productReveals = Array.prototype.slice.call(document.querySelectorAll(
    '.char-story__eyebrow, .char-story__grid > *, .char-detail-card, .char-final-cta > *'
  ));
  if (!reduceMotion && productReveals.length) {
    productReveals.forEach(function (el, i) {
      el.classList.add('char-reveal');
      el.style.setProperty('--char-delay', (i % 3) * 110 + 'ms');
    });
    if ('IntersectionObserver' in window) {
      var productObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          entry.target.classList.add('char-reveal--in');
          productObserver.unobserve(entry.target);
        });
      }, { threshold: 0.16, rootMargin: '0px 0px -5% 0px' });
      productReveals.forEach(function (el) { productObserver.observe(el); });
    } else {
      productReveals.forEach(function (el) { el.classList.add('char-reveal--in'); });
    }
  }

  var productStage = document.querySelector('.char-stage');
  if (productStage && !reduceMotion && finePointer) {
    productStage.addEventListener('pointermove', function (event) {
      var rect = productStage.getBoundingClientRect();
      var x = (event.clientX - rect.left) / rect.width - 0.5;
      var y = (event.clientY - rect.top) / rect.height - 0.5;
      productStage.style.setProperty('--stage-tilt-x', (x * 5).toFixed(2) + 'deg');
      productStage.style.setProperty('--stage-tilt-y', (y * -4).toFixed(2) + 'deg');
    });
    productStage.addEventListener('pointerleave', function () {
      productStage.style.setProperty('--stage-tilt-x', '0deg');
      productStage.style.setProperty('--stage-tilt-y', '0deg');
    });
  }
})();
