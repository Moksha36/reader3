(function (global) {
  "use strict";

  function savedWordAtPoint(highlights, clientX, clientY) {
    for (const highlight of highlights) {
      const rect = highlight.getBoundingClientRect();
      if (
        clientX >= rect.left
        && clientX <= rect.right
        && clientY >= rect.top
        && clientY <= rect.bottom
      ) {
        return highlight;
      }
    }
    return null;
  }

  function makeSavedWordInteractive(element, word) {
    element.tabIndex = 0;
    element.setAttribute("role", "button");
    element.setAttribute(
      "aria-label",
      `Show meaning and pronunciation for ${String(word || "saved word").trim()}`,
    );
  }

  function setTooltipVisible(tooltip, visible) {
    tooltip.setAttribute("aria-hidden", visible ? "false" : "true");
    if (visible) {
      tooltip.removeAttribute("inert");
    } else {
      tooltip.setAttribute("inert", "");
    }
  }

  global.Reader3ReaderInteractions = {
    savedWordAtPoint,
    makeSavedWordInteractive,
    setTooltipVisible,
  };
})(window);
