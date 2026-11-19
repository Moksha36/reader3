(function (global) {
  "use strict";

  function trimSurroundingPunctuation(value) {
    return String(value || "")
      .trim()
      .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");
  }

  function parseVocabularySelection(value) {
    const selectedText = trimSurroundingPunctuation(value);
    if (!selectedText) {
      return null;
    }

    const selectedWords = selectedText
      .split(/\s+/)
      .map(trimSurroundingPunctuation)
      .filter((word) => /^[\p{L}\p{N}][\p{L}\p{N}'’-]*$/u.test(word));
    if (selectedWords.length === 0) {
      return null;
    }

    const isPhrase = selectedWords.length > 1;
    return {
      selectedText,
      words: isPhrase ? [selectedText] : selectedWords,
      isPhrase,
    };
  }

  function normalizedText(value) {
    return String(value || "").replace(/\s+/g, " ").trim();
  }

  function contextMeaningFromFragment(fragment) {
    const blocks = Array.from(fragment.querySelectorAll(".meaning-block"));
    for (const block of blocks) {
      const title = normalizedText(block.querySelector(".meaning-title")?.textContent).toLocaleLowerCase();
      if (title === "context meaning") {
        return normalizedText(block.querySelector(".meaning-text")?.textContent);
      }
    }
    return blocks.length === 0 ? normalizedText(fragment.textContent) : "";
  }

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
    contextMeaningFromFragment,
    parseVocabularySelection,
    savedWordAtPoint,
    makeSavedWordInteractive,
    setTooltipVisible,
  };
})(window);
