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
