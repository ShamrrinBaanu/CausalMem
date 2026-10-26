import assert from "node:assert/strict";
import test from "node:test";
import { formatDisplayedAnswer, formatIngestionMessage } from "../src/utils/answerPresentation.ts";

const response = {
  citations: [],
  evidence: { states: [] },
};

const examples = [
  ["User prefers Java", "You prefer Java."],
  ["User prefers Python for programming", "You prefer Python for programming."],
  ["User's favorite color is blue", "Your favorite color is blue."],
  ["User enjoys running", "You enjoy running."],
  ["User's new goal is to learn system design", "Your new goal is to learn system design."],
  ["User completed their machine learning project", "You completed your machine learning project."],
  [
    "User switched from running to swimming due to a knee injury",
    "You switched from running to swimming due to a knee injury.",
  ],
];

for (const [stored, displayed] of examples) {
  test(`${stored} is presented in second person`, () => {
    assert.equal(formatDisplayedAnswer(stored, response), displayed);
  });
}

test("conflicting preferences remain an explicit conflict", () => {
  const answer =
    "Your stored preferences conflict: some records say you prefer Java, while others say you prefer Python.";
  assert.equal(formatDisplayedAnswer(answer, response), answer);
});

test("already-natural second-person wording is unchanged", () => {
  const answer = "You enjoy running.";
  assert.equal(formatDisplayedAnswer(answer, response), answer);
});

test("first-person programming preference summary is displayed in second person", () => {
  assert.equal(
    formatDisplayedAnswer("My current programming preference is Java", response),
    "Your current programming preference is Java.",
  );
});

for (const [input, expected] of [
  ["My favorite food is briyani", "Your favorite food is briyani."],
  ["My current city is Coimbatore", "Your current city is Coimbatore."],
  ["I prefer Java.", "You prefer Java."],
  ["I previously preferred Python.", "You previously preferred Python."],
]) {
  test(`first-person memory answer is shown naturally: ${input}`, () => {
    assert.equal(formatDisplayedAnswer(input, response), expected);
  });
}

test("quoted wording is left untouched", () => {
  const answer = 'The stored note says, “My favorite food is briyani.”';
  assert.equal(formatDisplayedAnswer(answer, response), answer);
});

test("successful ingestion shows only the concise state confirmation", () => {
  assert.equal(formatIngestionMessage({
    ingest_status: "state_recorded",
    states: [{ state_value: "briyani" }],
  }), "Memory stored in Mem0. State recorded: briyani.");
});

test("non-ingestable input is not presented as successful storage", () => {
  assert.match(formatIngestionMessage({
    ingest_status: "not_ingestable",
    states: [],
  }), /couldn't record/);
});

test("bare active preference values are displayed as a second-person answer only with their citation", () => {
  const evidence = {
    citations: [{ kind: "state", id: "java-state" }],
    evidence: {
      states: [{
        state_id: "java-state",
        state_type: "PREFERENCE",
        state_value: "Java",
        status: "ACTIVE",
        source_memory_ids: [],
      }],
    },
  };
  assert.equal(formatDisplayedAnswer("Java", evidence), "You prefer Java.");
  assert.equal(formatDisplayedAnswer("Java", response), "Java");
});

test("unrecognized text beginning with User is not blindly rewritten", () => {
  const answer = "User's brother prefers Java.";
  assert.equal(formatDisplayedAnswer(answer, response), answer);
});

test("causal answer retains its cause and becomes second person", () => {
  assert.equal(
    formatDisplayedAnswer(
      "User switched from running to swimming due to a knee injury",
      response,
    ),
    "You switched from running to swimming due to a knee injury.",
  );
});

test("Evidence Inspector response data stays separate from displayed wording", () => {
  const raw = "User enjoys running";
  const payload = {
    ...response,
    answer: raw,
    evidence: { states: [{ state_value: raw }] },
  };
  const displayed = formatDisplayedAnswer(payload.answer, payload);
  assert.equal(displayed, "You enjoy running.");
  assert.equal(payload.answer, raw);
  assert.equal(payload.evidence.states[0].state_value, raw);
});
