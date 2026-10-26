import type { AnswerResponse, IngestResponse } from "../types";

const SOURCE_ANNOTATION = /\[Sources:\s*([^\]\r\n]+)\]/g;
const SOURCE_REFERENCE = /^(memory|state|transition):([^\s,\[\]]+)$/;
const ABSTENTION_OR_UNCERTAINTY =
  /^I\b(?=.*\b(?:don't|do not|can't|cannot|couldn't|could not|won't|wouldn't|am not sure|am unsure|uncertain|unknown|not enough|no idea|might|may|possibly|perhaps|unable|can't tell|cannot tell)\b)/i;

function removeValidatedSourceAnnotations(
  answer: string,
  citations: AnswerResponse["citations"],
): string {
  const knownReferences = new Set(
    citations.map(({ kind, id }) => `${kind}:${id}`),
  );

  return answer
    .replace(SOURCE_ANNOTATION, (annotation, body: string) => {
      const references = body.split(",").map((reference) => reference.trim());
      const isStructuredCitation =
        references.length > 0 &&
        references.every(
          (reference) =>
            SOURCE_REFERENCE.test(reference) && knownReferences.has(reference),
        );
      return isStructuredCitation ? "" : annotation;
    })
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\s+([,.;!?])/g, "$1")
    .trim();
}

function hasCurrentMatchingGoal(answer: string, response: AnswerResponse): boolean {
  const normalizedAnswer = answer.toLocaleLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
  const now = Date.now();

  return response.evidence.states.some((state) => {
    if (state.state_type.toUpperCase() !== "GOAL" || state.status !== "ACTIVE") {
      return false;
    }
    const validFrom = Date.parse(state.valid_from);
    const validUntil = state.valid_until ? Date.parse(state.valid_until) : null;
    if (
      !Number.isFinite(validFrom) ||
      validFrom > now ||
      (validUntil !== null && (!Number.isFinite(validUntil) || now >= validUntil))
    ) {
      return false;
    }

    const normalizedValue = state.state_value
      .toLocaleLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, " ")
      .trim();
    return Boolean(normalizedValue && normalizedAnswer.includes(normalizedValue));
  });
}

const USER_POSSESSIVE_STATEMENT =
  /(^|[.!?]\s+)User(?:'s|’s)\s+(favorite\s+[^.!?]+)(?=$|[.!?])/gi;
const USER_ASSERTION =
  /(^|[.!?]\s+)User\s+(prefers|enjoys|likes|loves|wants|needs|uses|lives|works|completed|finished|achieved|switched|moved|lived)\b([^.!?]*)(?=$|[.!?])/gi;
const SECOND_PERSON_VERBS: Record<string, string> = {
  prefers: "prefer",
  enjoys: "enjoy",
  likes: "like",
  loves: "love",
  wants: "want",
  needs: "need",
  uses: "use",
  lives: "live",
  works: "work",
  completed: "completed",
  finished: "finished",
  achieved: "achieved",
  switched: "switched",
  moved: "moved",
  lived: "lived",
};
const MY_PROGRAMMING_PREFERENCE =
  /^My ((?:(?:current|desired|new|primary|main)\s+)*(?:programming(?:\s+language)?\s+)?preference is\b)/i;
const MY_PERSONAL_DETAIL =
  /^My ((?:favorite\s+[^.!?]+|(?:current|previous|past|former)\s+(?:programming\s+)?(?:preference|city|location|goal|project|hobby|sport|job|home))\s+(?:is|was|has been|used to be)\b)/i;

function makeStoredStatementsSecondPerson(answer: string): string {
  let natural = answer.replace(
    USER_POSSESSIVE_STATEMENT,
    (_match, sentenceStart: string, statement: string) =>
      `${sentenceStart}Your ${statement.trim()}`,
  );
  natural = natural.replace(
    USER_ASSERTION,
    (_match, sentenceStart: string, verb: string, remainder: string) => {
      const possessivesFixed = remainder.replace(/^\s+their\b/i, (match) =>
        match.replace(/their/i, "your"),
      );
      return `${sentenceStart}You ${SECOND_PERSON_VERBS[verb.toLowerCase()]}${possessivesFixed}`;
    },
  );
  if (natural !== answer && !/[.!?]$/.test(natural.trim())) {
    return `${natural.trim()}.`;
  }
  return natural;
}

function makeAnswerNatural(answer: string, response: AnswerResponse): string {
  if (ABSTENTION_OR_UNCERTAINTY.test(answer)) return answer;
  // Do not paraphrase quoted or conversational text; this layer only normalizes
  // unquoted, validated memory-answer phrasing.
  if (/[“”"‘’]/.test(answer)) return answer;

  const bareValue = answer.trim().replace(/[.!?]+$/, "").trim().toLocaleLowerCase();
  const activePreferenceMatches = response.evidence.states.filter((state) => {
    if (
      typeof state.state_type !== "string" ||
      state.state_type.toUpperCase() !== "PREFERENCE" ||
      state.status !== "ACTIVE" ||
      typeof state.state_value !== "string" ||
      !Array.isArray(state.source_memory_ids)
    ) {
      return false;
    }
    const normalizedValue = state.state_value.trim().toLocaleLowerCase();
    const hasSupportingCitation = response.citations.some(
      (citation) =>
        (citation.kind === "state" && citation.id === state.state_id) ||
        (citation.kind === "memory" && state.source_memory_ids.includes(citation.id)),
    );
    return Boolean(normalizedValue && bareValue === normalizedValue && hasSupportingCitation);
  });
  if (activePreferenceMatches.length === 1) {
    return `You prefer ${activePreferenceMatches[0].state_value}.`;
  }

  let natural = answer.replace(/^(?:User['’]s|My)\s+goal\s+is\b/i, "Your goal is");
  const newGoal = /^(?:User['’]s|My)\s+(new|current|primary|main)\s+goal\s+is\b/i.exec(answer);
  if (newGoal) {
    const goalIsCurrent = hasCurrentMatchingGoal(answer, response);
    const descriptor = goalIsCurrent ? "current" : newGoal[1].toLowerCase();
    natural = answer.replace(new RegExp(`^(?:User['’]s|My)\\s+${newGoal[1]}\\s+goal\\s+is\\b`, "i"),
      `Your ${descriptor} goal is`);
  }

  natural = natural.replace(MY_PROGRAMMING_PREFERENCE, "Your $1");
  natural = natural.replace(MY_PERSONAL_DETAIL, "Your $1");

  natural = makeStoredStatementsSecondPerson(natural);

  natural = natural.replace(
    /^I lived in (.+?) before([.!?]?)$/i,
    (_match, place: string, punctuation: string) =>
      `You previously lived in ${place}${punctuation || "."}`,
  );
  const secondPerson = natural.replace(/(^|[.!?]\s+)I\b/g, (match, sentenceStart: string, offset: number, whole: string) => {
    const sentence = whole.slice(offset + sentenceStart.length);
    return ABSTENTION_OR_UNCERTAINTY.test(sentence) ? match : `${sentenceStart}You`;
  });
  return secondPerson !== answer && !/[.!?]$/.test(secondPerson.trim())
    ? `${secondPerson.trim()}.`
    : secondPerson;
}

export function formatDisplayedAnswer(
  answer: string,
  response: AnswerResponse,
): string {
  return makeAnswerNatural(
    removeValidatedSourceAnnotations(answer, response.citations),
    response,
  );
}

export function formatIngestionMessage(response: IngestResponse): string {
  if (response.ingest_status === "not_ingestable") {
    return "I couldn't record that as a memory. Try phrasing it as a clear statement.";
  }
  if (response.ingest_status === "no_new_state") {
    return "Memory stored in Mem0. No new state was needed.";
  }

  const values = response.states
    .map(({ state_value }) => state_value.trim().replace(/[.!?]+$/, ""))
    .filter(Boolean);
  return values.length > 0
    ? `Memory stored in Mem0. State recorded: ${values.join(", ")}.`
    : "Memory stored in Mem0.";
}
