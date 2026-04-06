"use client";

import { useState, useEffect, Suspense } from "react";
import dynamic from "next/dynamic";
import { Clock, RefreshCw } from "lucide-react";
import type { TimelineEvent } from "@/lib/api";
import { api } from "@/lib/api";

const Timeline3D = dynamic(
  () => import("@/components/Timeline3D").then((m) => ({ default: m.Timeline3D })),
  { ssr: false, loading: () => <TimelineLoader /> },
);

function TimelineLoader() {
  return (
    <div className="flex items-center justify-center h-[600px]">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-sacred-gold border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-text-muted text-sm">Loading Sacred Timeline…</p>
      </div>
    </div>
  );
}

export default function TimelinePage() {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [agentId, setAgentId] = useState("http-server");
  const [error, setError] = useState<string | null>(null);

  async function fetchTimeline() {
    setLoading(true);
    setError(null);
    try {
      const data = await api.timeline(agentId, { limit: 200 });
      setEvents(data.events || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load timeline");
      setEvents(demoEvents());
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchTimeline();
  }, [agentId]);

  return (
    <div className="flex flex-col h-screen">
      <header className="flex items-center justify-between px-6 py-4 border-b border-void-light">
        <div className="flex items-center gap-3">
          <Clock className="w-5 h-5 text-sacred-gold" />
          <h1 className="text-xl font-semibold">Sacred Timeline</h1>
          <span className="text-xs bg-sacred-gold/20 text-sacred-gold px-2 py-0.5 rounded">
            {events.length} events
          </span>
        </div>
        <div className="flex items-center gap-3">
          <input
            type="text"
            value={agentId}
            onChange={(e) => setAgentId(e.target.value)}
            placeholder="Agent ID"
            className="bg-void-gray border border-void-light rounded-lg px-3 py-1.5 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-sacred-gold/50 w-48"
          />
          <button
            onClick={fetchTimeline}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-void-gray border border-void-light rounded-lg text-sm text-text-secondary hover:text-text-primary hover:border-sacred-gold/30 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>
      </header>

      {error && (
        <div className="mx-6 mt-3 px-4 py-2 bg-warning/10 border border-warning/30 rounded-lg text-sm text-warning">
          {error} — showing demo data
        </div>
      )}

      <div className="flex-1 relative">
        <Timeline3D events={events} />
      </div>
    </div>
  );
}

function demoEvents(): TimelineEvent[] {
  const now = Date.now();
  const types = ["write", "write", "write", "outcome", "webhook", "branch_created", "write", "write", "cross_agent_read", "branch_merged", "tag_created", "rollback"];
  return types.map((t, i) => ({
    id: `demo-${i}`,
    agent_id: "demo-agent",
    branch: i >= 5 && i <= 8 ? "experiment" : "main",
    event_type: t,
    summary: `Demo ${t.replace(/_/g, " ")} event #${i + 1}`,
    details: {},
    created_at: new Date(now - (types.length - i) * 3600000).toISOString(),
  }));
}
