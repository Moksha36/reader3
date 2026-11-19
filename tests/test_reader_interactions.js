const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "static", "reader_interactions.js"),
  "utf8",
);

const window = {};
vm.runInNewContext(source, { window });
const api = window.Reader3ReaderInteractions;

test("punctuation around a selected phrase does not discard its last word", () => {
  assert.deepEqual(
    plain(api.parseVocabularySelection("hold on.")),
    { selectedText: "hold on", words: ["hold on"], isPhrase: true },
  );
  assert.deepEqual(
    plain(api.parseVocabularySelection("“hold on”")),
    { selectedText: "hold on", words: ["hold on"], isPhrase: true },
  );
  assert.deepEqual(
    plain(api.parseVocabularySelection("meticulous,")),
    { selectedText: "meticulous", words: ["meticulous"], isPhrase: false },
  );
});

test("structured output without context does not expose examples as meaning", () => {
  const examplesOnly = fakeFragment(
    [fakeMeaningBlock("Examples", "First example. Second example. Third example.")],
    "Examples First example. Second example. Third example.",
  );
  const context = fakeFragment(
    [
      fakeMeaningBlock("Pronunciation", "/test/"),
      fakeMeaningBlock("Context meaning", "  Meaning in this sentence.  "),
      fakeMeaningBlock("Examples", "Three examples."),
    ],
    "/test/ Context meaning Meaning in this sentence. Examples Three examples.",
  );
  const unstructuredLegacy = fakeFragment([], "  A legacy plain-text meaning.  ");

  assert.equal(api.contextMeaningFromFragment(examplesOnly), "");
  assert.equal(api.contextMeaningFromFragment(context), "Meaning in this sentence.");
  assert.equal(api.contextMeaningFromFragment(unstructuredLegacy), "A legacy plain-text meaning.");
});

function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

function fakeMeaningBlock(title, text) {
  return {
    querySelector(selector) {
      if (selector === ".meaning-title") {
        return { textContent: title };
      }
      if (selector === ".meaning-text") {
        return { textContent: text };
      }
      return null;
    },
  };
}

function fakeFragment(blocks, textContent) {
  return {
    textContent,
    querySelectorAll(selector) {
      return selector === ".meaning-block" ? blocks : [];
    },
  };
}

test("PDF saved words can be resolved from pointer coordinates", () => {
  const first = highlight({ left: 10, top: 20, right: 60, bottom: 40 });
  const second = highlight({ left: 70, top: 20, right: 120, bottom: 40 });

  assert.equal(api.savedWordAtPoint([first, second], 80, 30), second);
  assert.equal(api.savedWordAtPoint([first, second], 200, 30), null);
});

test("saved words are keyboard-focusable controls", () => {
  const element = fakeElement();

  api.makeSavedWordInteractive(element, "example");

  assert.equal(element.tabIndex, 0);
  assert.equal(element.getAttribute("role"), "button");
  assert.equal(element.getAttribute("aria-label"), "Show meaning and pronunciation for example");
});

test("hidden tooltip is removed from keyboard interaction", () => {
  const tooltip = fakeElement();

  api.setTooltipVisible(tooltip, true);
  assert.equal(tooltip.hasAttribute("inert"), false);
  assert.equal(tooltip.getAttribute("aria-hidden"), "false");

  api.setTooltipVisible(tooltip, false);
  assert.equal(tooltip.hasAttribute("inert"), true);
  assert.equal(tooltip.getAttribute("aria-hidden"), "true");
});

function highlight(rect) {
  return { getBoundingClientRect: () => rect };
}

function fakeElement() {
  const attributes = new Map();
  return {
    tabIndex: -1,
    setAttribute(name, value = "") {
      attributes.set(name, String(value));
    },
    removeAttribute(name) {
      attributes.delete(name);
    },
    getAttribute(name) {
      return attributes.get(name) ?? null;
    },
    hasAttribute(name) {
      return attributes.has(name);
    },
  };
}
