function wire() {
  const homepage = "https://proteus-framework.org/";

  const logo = document.querySelector(".md-header__button.md-logo");
  let docsHome = location.origin + "/";
  if (logo) {
    // The theme points the logo at this site's own home page. Remember that
    // before repointing the logo at the framework home, so the title can use
    // it and no page has to know its own name.
    if (!logo.dataset.docsHome) logo.dataset.docsHome = logo.href;
    docsHome = logo.dataset.docsHome;
    logo.href = homepage;
  }

  const title = document.querySelector(".md-header__title[data-md-component='header-title']");
  if (title && !title.dataset.titleWired) {
    title.dataset.titleWired = "1";
    title.style.cursor = "pointer";
    title.addEventListener("click", (e) => {
      if (e.target.closest("a, button, input, label")) return;
      window.location.assign(docsHome);
    }, true);
  }
}

document.addEventListener("DOMContentLoaded", wire);
if (window.document$?.subscribe) window.document$.subscribe(wire);
