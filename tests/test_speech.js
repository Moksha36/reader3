const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const speechSource = fs.readFileSync(
  path.join(__dirname, "..", "static", "speech.js"),
  "utf8",
);

function loadSpeech({ supported = true, voices = [] } = {}) {
  const spoken = [];
  let cancelCount = 0;
  const speechSynthesis = {
    cancel() {
      cancelCount += 1;
    },
    getVoices() {
      return voices;
    },
    speak(utterance) {
      spoken.push(utterance);
    },
  };

  class SpeechSynthesisUtterance {
    constructor(text) {
      this.text = text;
      this.lang = "";
      this.rate = 1;
      this.voice = null;
    }
  }

  const window = {};
  if (supported) {
    window.speechSynthesis = speechSynthesis;
    window.SpeechSynthesisUtterance = SpeechSynthesisUtterance;
  }

  vm.runInNewContext(speechSource, { window });
  return {
    api: window.Reader3Speech,
    spoken,
    cancelCount: () => cancelCount,
  };
}

test("unsupported browsers report unavailable and do not speak", () => {
  const { api } = loadSpeech({ supported: false });

  assert.equal(api.isSupported(), false);
  assert.equal(api.speak("example"), false);
});

test("blank text does not create an utterance", () => {
  const { api, spoken } = loadSpeech();

  assert.equal(api.speak("   "), false);
  assert.equal(spoken.length, 0);
});

test("speaking cancels current speech and prefers a local English voice", () => {
  const remoteEnglish = { name: "Remote English", lang: "en-GB", localService: false };
  const localFrench = { name: "Local French", lang: "fr-FR", localService: true };
  const localEnglish = { name: "Local English", lang: "en-US", localService: true };
  const { api, spoken, cancelCount } = loadSpeech({
    voices: [remoteEnglish, localFrench, localEnglish],
  });

  assert.equal(api.speak("example"), true);
  assert.equal(cancelCount(), 1);
  assert.equal(spoken.length, 1);
  assert.equal(spoken[0].text, "example");
  assert.equal(spoken[0].lang, "en-US");
  assert.equal(spoken[0].rate, 0.85);
  assert.equal(spoken[0].voice, localEnglish);
});

test("speaking works with the browser default when voices are not loaded", () => {
  const { api, spoken } = loadSpeech({ voices: [] });

  assert.equal(api.speak("example"), true);
  assert.equal(spoken.length, 1);
  assert.equal(spoken[0].voice, null);
});

test("stop cancels current speech", () => {
  const { api, cancelCount } = loadSpeech();

  api.stop();

  assert.equal(cancelCount(), 1);
});
