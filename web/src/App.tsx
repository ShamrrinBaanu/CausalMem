import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "./api";
import type { ChatMessage, ChatResponse, Config, View } from "./types";
import { ErrorBanner, Icon, type IconName } from "./components/ui";
import ChatPage from "./pages/ChatPage";
import MemoryPage from "./pages/MemoryPage";
import TimelinePage from "./pages/TimelinePage";
import EvidencePage from "./pages/EvidencePage";
import DiagnosticsPage from "./pages/DiagnosticsPage";
import SettingsPage from "./pages/SettingsPage";

const NAV_ITEMS: Array<{ view: View; label: string; icon: IconName; hint: string }> = [
  { view: "chat", label: "Chat", icon: "chat", hint: "Ask and ingest" },
  { view: "memories", label: "Memory Explorer", icon: "memories", hint: "Mem0 and states" },
  { view: "timeline", label: "Timeline", icon: "timeline", hint: "State history" },
  { view: "evidence", label: "Evidence Inspector", icon: "evidence", hint: "Citations and gates" },
  { view: "diagnostics", label: "Diagnostics", icon: "diagnostics", hint: "Store health" },
  { view: "settings", label: "Settings", icon: "settings", hint: "Namespace and model" },
];

const VIEW_TITLES: Record<View, { title: string; eyebrow: string }> = {
  chat: { title: "Conversation", eyebrow: "Memory-grounded dialogue" },
  memories: { title: "Memory explorer", eyebrow: "Stored records" },
  timeline: { title: "Timeline", eyebrow: "State and transition history" },
  evidence: { title: "Evidence inspector", eyebrow: "Response provenance" },
  diagnostics: { title: "Diagnostics", eyebrow: "Operational health" },
  settings: { title: "Settings", eyebrow: "Workspace configuration" },
};

function readStoredNamespace(): string | null {
  try {
    return window.localStorage.getItem("causalmem.namespace");
  } catch {
    return null;
  }
}

export default function App() {
  const [view, setView] = useState<View>("chat");
  const [config, setConfig] = useState<Config | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [userId, setUserIdState] = useState(() => readStoredNamespace() ?? "default");
  const [chats, setChats] = useState<Record<string, ChatMessage[]>>({});
  const [lastResponses, setLastResponses] = useState<Record<string, ChatResponse>>({});

  useEffect(() => {
    let cancelled = false;
    api
      .getConfig()
      .then((loaded) => {
        if (cancelled) return;
        setConfig(loaded);
        if (!readStoredNamespace() && loaded.default_user) {
          setUserIdState(loaded.default_user);
        }
      })
      .catch((caught) => {
        if (cancelled) return;
        setConfigError(
          caught instanceof ApiError
            ? caught.message
            : "Could not reach the CausalMem API. Start the FastAPI server on port 8000.",
        );
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function setUserId(next: string) {
    setUserIdState(next);
    try {
      window.localStorage.setItem("causalmem.namespace", next);
    } catch {
      // Persistence is optional; the in-memory namespace still applies.
    }
  }

  const messages = chats[userId] ?? [];
  const lastResponse = lastResponses[userId] ?? null;

  const setMessages = useMemo(
    () => (updater: (previous: ChatMessage[]) => ChatMessage[]) => {
      setChats((previous) => ({
        ...previous,
        [userId]: updater(previous[userId] ?? []),
      }));
    },
    [userId],
  );

  const setResponse = (response: ChatResponse) => {
    setLastResponses((previous) => ({ ...previous, [userId]: response }));
  };

  const activeTitle = VIEW_TITLES[view];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">
            <span>CM</span>
          </span>
          <div className="brand-copy">
            <strong>CausalMem</strong>
            <span>Memory research console</span>
          </div>
        </div>

        <nav className="nav" aria-label="Primary">
          <span className="nav-label">Workspace</span>
          {NAV_ITEMS.map((item) => (
            <button
              key={item.view}
              type="button"
              className={`nav-item ${view === item.view ? "active" : ""}`}
              onClick={() => setView(item.view)}
              aria-current={view === item.view ? "page" : undefined}
            >
              <span className="nav-icon"><Icon name={item.icon} size={18} /></span>
              <span className="nav-copy">
                <strong>{item.label}</strong>
                <span>{item.hint}</span>
              </span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="status-line">
            <span className="status-dot" />
            {config ? "Local service ready" : "Connecting to local service"}
          </div>
          <span className="build-note">Evidence-first build · no cloud dependency</span>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="topbar-title">
            <span className="eyebrow">{activeTitle.eyebrow}</span>
            <h1>{activeTitle.title}</h1>
          </div>
          <div className="topbar-meta">
            <span className="meta-chip" title="Active memory namespace">
              <Icon name="shield" size={14} />
              {userId}
            </span>
            <span className="meta-chip" title="Configured provider and model">
              <Icon name="database" size={14} />
              {config ? `${config.provider} · ${config.model}` : "backend"}
            </span>
            {lastResponse && (
              <button
                type="button"
                className="response-chip"
                onClick={() => setView(lastResponse.response_type === "answer" ? "evidence" : "diagnostics")}
                title="Open the latest response trace"
              >
                <span className={`response-dot ${lastResponse.response_type}`} />
                {lastResponse.response_type === "answer"
                  ? lastResponse.diagnostics.answer_reason.replaceAll("_", " ")
                  : lastResponse.ingest_status.replaceAll("_", " ")}
                <span className="response-time">{lastResponse.timings.total_ms.toFixed(0)} ms</span>
              </button>
            )}
          </div>
        </header>

        {configError && (
          <div className="global-banner">
            <ErrorBanner message={configError} onDismiss={() => setConfigError(null)} />
          </div>
        )}

        {view === "chat" ? (
          <div className="view view-chat">
            <ChatPage
              key={userId}
              userId={userId}
              messages={messages}
              setMessages={setMessages}
              onResponse={setResponse}
              onInspectEvidence={() => setView("evidence")}
            />
          </div>
        ) : (
          <div className="view view-scroll">
            {view === "memories" && <MemoryPage userId={userId} />}
            {view === "timeline" && <TimelinePage userId={userId} />}
            {view === "evidence" && <EvidencePage response={lastResponse} userId={userId} />}
            {view === "diagnostics" && <DiagnosticsPage userId={userId} lastResponse={lastResponse} />}
            {view === "settings" && <SettingsPage userId={userId} setUserId={setUserId} config={config} />}
          </div>
        )}
      </main>
    </div>
  );
}
