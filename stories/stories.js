const grid = document.getElementById('issue-grid');
const issueCount = document.getElementById('issue-count');
const dialog = document.getElementById('story-reader');
const readerImage = document.getElementById('reader-image');
const readerTitle = document.getElementById('reader-title');
const readerIssueLabel = document.getElementById('reader-issue-label');
const readerPageCount = document.getElementById('reader-page-count');
const readerProgress = document.getElementById('reader-progress-bar');
const closeButton = document.getElementById('reader-close');
const prevButtons = [document.getElementById('reader-prev'), document.getElementById('reader-prev-label')];
const nextButtons = [document.getElementById('reader-next'), document.getElementById('reader-next-label')];
const readerStage = document.getElementById('reader-stage');

let catalog = [];
let activeIssue = null;
let activeManifest = null;
let activePage = 0;
let opener = null;
let touchStartX = null;

function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderIssueCard(issue) {
  const article = make('article', 'issue-card');
  const button = make('button', 'issue-card__open');
  button.type = 'button';
  button.setAttribute('aria-label', `Read Issue ${issue.number}: ${issue.title}`);

  const cover = make('span', 'issue-card__cover');
  const image = make('img');
  image.src = issue.cover;
  image.alt = `Preview artwork for REDPOLY Issue ${issue.number}: ${issue.title}`;
  image.loading = 'lazy';
  cover.append(image, make('span', 'issue-card__number', `ISSUE ${issue.number}`), make('span', 'issue-card__cover-label', 'THE ORIGIN ISSUE'));

  const body = make('span', 'issue-card__body');
  body.append(make('span', 'issue-card__eyebrow', `REDPOLY COMICS · NO. ${issue.number}`));
  body.append(make('h3', 'issue-card__title', issue.title));
  body.append(make('span', 'issue-card__description', issue.subtitle));
  const meta = make('span', 'issue-card__meta');
  meta.append(make('span', '', `${issue.pageCount} COMIC PAGES`), make('span', '', `${issue.panelCount} PANELS`));
  body.append(meta);
  const cta = make('span', 'issue-card__cta');
  cta.append(make('span', '', issue.status || `READ ISSUE ${issue.number}`), make('b', '', '↗'));
  body.append(cta);
  button.append(cover, body);
  button.addEventListener('click', () => openIssue(issue, 0, button));
  article.append(button);
  return article;
}

function renderCatalog(items) {
  grid.replaceChildren();
  if (!items.length) {
    grid.append(make('p', 'shelf-status', 'A new story is on its way. Check back after the next splash.'));
    issueCount.textContent = 'MORE ISSUES COMING SOON';
    return;
  }
  items.forEach(issue => grid.append(renderIssueCard(issue)));
  issueCount.textContent = items.length === 1 ? 'ISSUE 01 · THE FIRST SPLASH STARTS HERE' : `${items.length} ISSUES · PICK A STORY`;
}

function updateShareUrl() {
  const url = new URL(window.location.href);
  url.searchParams.set('issue', activeIssue.id);
  url.searchParams.set('page', String(activePage + 1));
  history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
}

function renderPage() {
  if (!activeManifest) return;
  const page = activeManifest.pages[activePage];
  const current = activePage + 1;
  const total = activeManifest.pages.length;
  const pageNumber = String(current).padStart(2, '0');

  readerImage.src = page.image;
  readerImage.alt = page.alt || `${activeManifest.title}, page ${current}`;
  readerPageCount.textContent = `PAGE ${pageNumber} / ${String(total).padStart(2, '0')}`;
  readerProgress.style.width = `${(current / total) * 100}%`;
  prevButtons.forEach(button => { button.disabled = activePage === 0; });
  nextButtons.forEach(button => { button.disabled = activePage === total - 1; });
}

async function openIssue(issue, pageIndex = 0, trigger = null) {
  if (!issue) return;
  opener = trigger || opener;
  try {
    const response = await fetch(issue.manifest);
    if (!response.ok) throw new Error(`Issue data returned ${response.status}`);
    activeManifest = await response.json();
    activeIssue = issue;
    activePage = Math.min(Math.max(pageIndex, 0), activeManifest.pages.length - 1);
    readerTitle.textContent = activeManifest.title;
    readerIssueLabel.textContent = `REDPOLY COMICS · ISSUE ${activeManifest.number}`;
    renderPage();
    if (!dialog.open) dialog.showModal();
    updateShareUrl();
    closeButton.focus();
  } catch (error) {
    console.error('Could not open comic issue:', error);
    grid.prepend(make('p', 'shelf-status shelf-status--error', 'This issue did not load. Please try again in a moment.'));
  }
}

function goToPage(index) {
  if (!activeManifest || index < 0 || index >= activeManifest.pages.length) return;
  activePage = index;
  renderPage();
  updateShareUrl();
}

function closeReader() {
  if (dialog.open) dialog.close();
  const url = new URL(window.location.href);
  url.searchParams.delete('issue');
  url.searchParams.delete('page');
  history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
  if (opener && document.contains(opener)) opener.focus();
}

prevButtons.forEach(button => button.addEventListener('click', () => goToPage(activePage - 1)));
nextButtons.forEach(button => button.addEventListener('click', () => goToPage(activePage + 1)));
closeButton.addEventListener('click', closeReader);
dialog.addEventListener('cancel', event => { event.preventDefault(); closeReader(); });
dialog.addEventListener('click', event => { if (event.target === dialog) closeReader(); });
dialog.addEventListener('keydown', event => {
  if (event.key === 'ArrowRight') { event.preventDefault(); goToPage(activePage + 1); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); goToPage(activePage - 1); }
});
readerStage.addEventListener('pointerdown', event => { touchStartX = event.clientX; }, { passive: true });
readerStage.addEventListener('pointerup', event => {
  if (touchStartX === null) return;
  const delta = event.clientX - touchStartX;
  touchStartX = null;
  if (Math.abs(delta) > 60) goToPage(activePage + (delta < 0 ? 1 : -1));
}, { passive: true });

async function startStories() {
  try {
    const response = await fetch('/comics/catalog.json');
    if (!response.ok) throw new Error(`Catalog returned ${response.status}`);
    catalog = await response.json();
    renderCatalog(catalog);
    const params = new URLSearchParams(window.location.search);
    const requestedIssue = catalog.find(issue => issue.id === params.get('issue'));
    if (requestedIssue) {
      const requestedPage = Math.max(0, Number(params.get('page') || 1) - 1);
      openIssue(requestedIssue, requestedPage);
    }
  } catch (error) {
    console.error('Could not load comic catalog:', error);
    grid.replaceChildren(make('p', 'shelf-status shelf-status--error', 'The story shelf could not load. Please refresh to try again.'));
  }
}

startStories();
