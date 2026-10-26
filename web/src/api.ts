import type {
  ChatResponse,
  Config,
  DiagnosticSnapshot,
  MemoryRecord,
  StateRecord,
  TransitionRecord,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const body = (await response.json()) as { detail?: string | Array<{ msg?: string }> };
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body.detail) && body.detail[0]?.msg) {
        detail = body.detail[0].msg;
      }
    } catch {
      // Keep the status-based fallback; never surface raw server output.
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

function namespace(userId: string): string {
  return encodeURIComponent(userId.trim());
}

export const api = {
  getConfig: () => request<Config>("/api/config"),

  getMemories: (userId: string) =>
    request<{ user_id: string; memories: MemoryRecord[] }>(
      `/api/memories?user_id=${namespace(userId)}`,
    ),

  getStates: (userId: string, includeInactive = true) =>
    request<{ user_id: string; states: StateRecord[] }>(
      `/api/states?user_id=${namespace(userId)}&include_inactive=${includeInactive}`,
    ),

  getTransitions: (userId: string) =>
    request<{ user_id: string; transitions: TransitionRecord[] }>(
      `/api/transitions?user_id=${namespace(userId)}`,
    ),

  getTimeline: (userId: string) =>
    request<{ user_id: string; states: StateRecord[]; transitions: TransitionRecord[] }>(
      `/api/timeline?user_id=${namespace(userId)}`,
    ),

  getDiagnostics: (userId: string) =>
    request<DiagnosticSnapshot>(`/api/diagnostics?user_id=${namespace(userId)}`),

  chat: (userId: string, message: string, requestedTimestamp?: string) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        user_id: userId,
        message,
        requested_timestamp: requestedTimestamp || null,
      }),
    }),
};
