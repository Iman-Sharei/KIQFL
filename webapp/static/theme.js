(function () {
  var KEY = "kiqfl-theme";

  function current() {
    return document.documentElement.getAttribute("data-theme") || "dark";
  }

  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(KEY, theme);
    var btn = document.getElementById("themeToggle");
    if (!btn) return;
    if (theme === "light") {
      btn.textContent = "Dark mode";
      btn.setAttribute("aria-label", "Switch to dark theme");
    } else {
      btn.textContent = "Light mode";
      btn.setAttribute("aria-label", "Switch to light theme");
    }
  }

  window.toggleKiqflTheme = function () {
    apply(current() === "dark" ? "light" : "dark");
  };

  document.addEventListener("DOMContentLoaded", function () {
    apply(current());
  });
})();
