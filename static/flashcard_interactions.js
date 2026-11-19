(function (global) {
  "use strict";

  function shouldHandleGlobalShortcut(event) {
    const tagName = event && event.target && event.target.tagName;
    const key = event && event.key;
    return !(tagName === "BUTTON" && (key === " " || key === "Enter"));
  }

  function syncFaceInteractivity(card, isFlipped) {
    const front = card.querySelector(".card-front");
    const back = card.querySelector(".card-back");
    if (!front || !back) {
      return;
    }

    if (isFlipped) {
      front.setAttribute("inert", "");
      back.removeAttribute("inert");
    } else {
      front.removeAttribute("inert");
      back.setAttribute("inert", "");
    }
  }

  global.Reader3FlashcardInteractions = {
    shouldHandleGlobalShortcut,
    syncFaceInteractivity,
  };
})(window);
