import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api";
import type { ChatMessage, ChatResponse } from "../types";
import { formatDisplayedAnswer, formatIngestionMessage } from "../utils/answerPresentation";
import { Badge, ErrorBanner, Icon, Spinner, reasonLabel } from "../components/ui";

interface ChatPageProps {
  userId: string;
  messages: ChatMessage[];
  setMessages: (updater: (previous: ChatMessage[]) => ChatMessage[]) => void;
  onResponse: (response: ChatResponse) => void;
  onInspectEvidence: () => void;
}

const starterPrompts = [
  "What do you remember about me?",
  "Where do I live now?",
  "What did I complete?",
  "What changed in my history?",
];

const QUESTION_START =
  /^(?:what|who|where|when|why|how|which|do|does|did|is|are|was|were|can|could|should|would|will|tell|show|list)\b/i;

export default function ChatPage({
  userId,
  messages,
  setMessages,
  onResponse,
  onInspectEvidence,
}: ChatPageProps) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [messages, loading]);

  const composerHint = useMemo(
    () => (loading ? "Retrieving evidence and validating claims…" : "Enter to send · Shift+Enter for a new line"),
    [loading],
  );
  const trimmedInput = input.trim();
  const looksLikeQuestion =
    trimmedInput.endsWith("?") || QUESTION_START.test(trimmedInput);
  const mode = trimmedInput
    ? looksLikeQuestion
      ? { label: "Ask", hint: "Retrieve and validate" }
      : { label: "Remember", hint: "Extract typed state" }
    : { label: "Ask or remember", hint: "Questions answer; statements write" };

  async function sendMessage(message: string) {
    const trimmed = message.trim();
    if (!trimmed || loading) return;
    setInput("");
    setError(null);
    setMessages((previous) => [
      ...previous,
      { role: "user", content: trimmed },
    ]);
    setLoading(true);
    try {
      const response = await api.chat(userId, trimmed);
      onResponse(response);
      setMessages((previous) => [
        ...previous,
        { role: "assistant", content: response.answer, response },
      ]);
    } catch (caught) {
      const detail =
        caught instanceof ApiError
          ? caught.message
          : "The request could not be completed. Check the backend and local provider.";
      setError(detail);
      setMessages((previous) => [
        ...previous,
        { role: "assistant", content: detail, error: detail },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    void sendMessage(input);
  }

  return (
    <div className="chat-layout">
      <div className="chat-stream">
        {messages.length === 0 && !loading && (
          <div className="chat-intro">
            <span className="intro-mark">
              <Icon name="spark" size={24} />
            </span>
            <h1>Ask your memory</h1>
            <p>
              CausalMem retrieves Mem0 memories, typed state history, and recorded transitions.
              Every answer is validated against the exact evidence bundle before it is shown.
            </p>
            <div className="starter-grid">
              {starterPrompts.map((prompt) => (
                <button
                  type="button"
                  className="starter"
                  key={prompt}
                  onClick={() => void sendMessage(prompt)}
                >
                  <span>{prompt}</span>
                  <Icon name="arrow" size={15} />
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) =>
          message.role === "user" ? (
            <div className="message message-user" key={`user-${index}`}>
              <div className="message-meta">You</div>
              <div className="user-bubble">{message.content}</div>
            </div>
          ) : (
            <AssistantMessage
              key={`assistant-${index}`}
              message={message}
              onInspectEvidence={onInspectEvidence}
            />
          ),
        )}

        {loading && (
          <div className="message message-assistant">
            <div className="message-meta">
              <span className="assistant-dot" /> CausalMem
            </div>
            <div className="assistant-bubble loading-bubble">
              <Spinner label="Building evidence bundle" />
              <div className="skeleton-line" />
              <div className="skeleton-line short" />
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="composer-wrap">
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void sendMessage(input);
              }
            }}
            placeholder="Ask about your memory, or tell CausalMem something new…"
            rows={2}
            maxLength={4000}
            disabled={loading}
            aria-label="Message"
          />
          <button type="submit" className="send-button" disabled={loading || !input.trim()}>
            {loading ? "Working" : "Send"}
            <Icon name="arrow" size={16} />
          </button>
        </form>
        <div className="composer-hint">
          <span
            className={`mode-chip ${
              !trimmedInput ? "mode-idle" : looksLikeQuestion ? "mode-ask" : "mode-remember"
            }`}
          >
            <Icon name={!trimmedInput ? "spark" : looksLikeQuestion ? "search" : "layers"} size={12} />
            {mode.label}
            <em>{mode.hint}</em>
          </span>
          <span>{composerHint}</span>
          <span className="namespace-hint">Namespace: {userId}</span>
        </div>
      </div>
    </div>
  );
}

function AssistantMessage({
  message,
  onInspectEvidence,
}: {
  message: ChatMessage;
  onInspectEvidence: () => void;
}) {
  const response = message.response;
  const answer = response?.response_type === "answer"
    ? formatDisplayedAnswer(response.answer, response)
    : response?.response_type === "ingest"
      ? formatIngestionMessage(response)
      : message.content;
  const isAnswer = response?.response_type === "answer";
  const isIngest = response?.response_type === "ingest";
  const isIngestFailure = isIngest && response.ingest_status === "not_ingestable";
  const isAbstention = isAnswer && answer.startsWith("I don't have enough information");
  const diagnostics = response?.response_type === "answer" ? response.diagnostics : null;

  return (
    <div className="message message-assistant">
      <div className="message-meta">
        <span className="assistant-dot" /> CausalMem
      </div>
      <div className={`assistant-bubble ${message.error || isIngestFailure ? "bubble-error" : ""}`}>
        {message.error ? (
          <ErrorBanner message={message.error} />
        ) : (
          <>
            <div className="answer-text">{answer}</div>
            {isAbstention && (
              <div className="abstention-note">
                <Icon name="shield" size={15} />
                Safe abstention: the retrieved evidence did not support a validated claim.
              </div>
            )}
            {response && isAnswer && (
              <div className="answer-footer">
                <div className="citation-row">
                  {response.citations.length > 0 ? (
                    response.citations.map((citation) => (
                      <Badge
                        key={`${citation.kind}-${citation.id}`}
                        tone={citation.kind === "transition" ? "warning" : citation.kind === "state" ? "accent" : "violet"}
                        className="citation-badge"
                        title={`${citation.kind}:${citation.id}`}
                      >
                        <Icon name={citation.kind === "transition" ? "branch" : citation.kind === "state" ? "layers" : "database"} size={12} />
                        {citation.kind}:{citation.id.slice(0, 10)}…
                      </Badge>
                    ))
                  ) : (
                    <span className="muted-small">No citations</span>
                  )}
                </div>
                <div className="answer-stats">
                  {diagnostics && (
                    <>
                      <span>{reasonLabel(diagnostics.answer_reason)}</span>
                      <span>{diagnostics.accepted_claim_count} accepted</span>
                      <span>{diagnostics.rejected_claim_count} rejected</span>
                    </>
                  )}
                  {response.timings.retrieval_ms !== undefined && (
                    <span>{response.timings.retrieval_ms.toFixed(0)} ms retrieval</span>
                  )}
                  <span>{response.timings.total_ms.toFixed(0)} ms</span>
                  {response.request_id && (
                    <span className="mono request-id" title={response.request_id}>
                      {response.request_id}
                    </span>
                  )}
                  {isAnswer && (
                    <button
                      type="button"
                      className="inline-link-button"
                      onClick={onInspectEvidence}
                    >
                      Inspect evidence
                      <Icon name="arrow" size={13} />
                    </button>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
