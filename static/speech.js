(function initializeReader3Speech(windowObject) {
    "use strict";

    function isSupported() {
        return Boolean(
            windowObject.speechSynthesis
            && windowObject.SpeechSynthesisUtterance
        );
    }

    function preferredVoice() {
        const voices = windowObject.speechSynthesis.getVoices?.() || [];
        return voices.find((voice) => voice.localService && /^en(?:-|$)/i.test(voice.lang || ""))
            || voices.find((voice) => /^en(?:-|$)/i.test(voice.lang || ""))
            || null;
    }

    function speak(text) {
        const spokenText = String(text || "").trim();
        if (!isSupported() || !spokenText) {
            return false;
        }

        windowObject.speechSynthesis.cancel();
        const utterance = new windowObject.SpeechSynthesisUtterance(spokenText);
        utterance.lang = "en-US";
        utterance.rate = 0.85;
        const voice = preferredVoice();
        if (voice) {
            utterance.voice = voice;
        }
        windowObject.speechSynthesis.speak(utterance);
        return true;
    }

    function stop() {
        if (isSupported()) {
            windowObject.speechSynthesis.cancel();
        }
    }

    windowObject.Reader3Speech = Object.freeze({
        isSupported,
        speak,
        stop,
    });
}(window));
