const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "static", "flashcard_interactions.js"),
  "utf8",
);

const window = {};
vm.runInNewContext(source, { window });
const api = window.Reader3FlashcardInteractions;

test("speaker buttons keep navigation shortcuts while retaining native activation", () => {
  assert.equal(api.shouldHandleGlobalShortcut({ key: "ArrowRight", target: { tagName: "BUTTON" } }), true);
  assert.equal(api.shouldHandleGlobalShortcut({ key: "a", target: { tagName: "BUTTON" } }), true);
  assert.equal(api.shouldHandleGlobalShortcut({ key: " ", target: { tagName: "BUTTON" } }), false);
  assert.equal(api.shouldHandleGlobalShortcut({ key: "Enter", target: { tagName: "BUTTON" } }), false);
});

test("only the visible flashcard face is interactive", () => {
  const front = fakeFace();
  const back = fakeFace();
  const card = {
    querySelector(selector) {
      return selector === ".card-front" ? front : back;
    },
  };

  api.syncFaceInteractivity(card, false);
  assert.equal(front.hasAttribute("inert"), false);
  assert.equal(back.hasAttribute("inert"), true);

  api.syncFaceInteractivity(card, true);
  assert.equal(front.hasAttribute("inert"), true);
  assert.equal(back.hasAttribute("inert"), false);
});

function fakeFace() {
  const attributes = new Set();
  return {
    setAttribute(name) {
      attributes.add(name);
    },
    removeAttribute(name) {
      attributes.delete(name);
    },
    hasAttribute(name) {
      return attributes.has(name);
    },
  };
}
