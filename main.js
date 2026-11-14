/* CHOMPKIT interactions and lightweight motion. */
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

  // Add to cart: quick button confirmation + toast.
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

  // Reveal cards and play the torn-edge wipe once as the limited section enters view.
  var revealEls = document.querySelectorAll('.p-card, .c-card, .limited__center, .limited__right');
  var limitedSection = document.querySelector('.limited');
  revealEls.forEach(function (el) { el.setAttribute('data-reveal', ''); });
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        if (entry.target === limitedSection) {
          entry.target.classList.add('is-visible');
        } else {
          entry.target.classList.add('in');
        }
        io.unobserve(entry.target);
      });
    }, { threshold: 0.12 });
    revealEls.forEach(function (el) { io.observe(el); });
    if (limitedSection) io.observe(limitedSection);
  } else {
    revealEls.forEach(function (el) { el.classList.add('in'); });
    if (limitedSection) limitedSection.classList.add('is-visible');
  }
})();
