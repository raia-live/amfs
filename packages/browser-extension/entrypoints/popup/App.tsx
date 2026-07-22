import { useEffect, useMemo, useState } from "react";
import { browser } from "wxt/browser";
import { track } from "@/utils/analytics";
import {
  CONNECT_URL,
  CRITICAL_THRESHOLD,
  DASHBOARD_URL,
  OPS_PER_SAVE,
  UPGRADE_URL,
  WARN_THRESHOLD,
} from "@/utils/config";
import {
  getLastSave,
  getRoomsState,
  getSettings,
  getUsage,
  updateSettings,
} from "@/utils/storage";
import type {
  LastSave,
  RoomsState,
  SaveOutcome,
  Settings,
  UsageInfo,
} from "@/utils/types";

type View = "loading" | "connect" | "main" | "quota";

export default function App() {
  const [view, setView] = useState<View>("loading");
  const [settings, setSettings] = useState<Settings | null>(null);
  const [usage, setUsage] = useState<UsageInfo | null>(null);
  const [lastSave, setLastSaveState] = useState<LastSave | null>(null);
  const [roomsState, setRoomsState] = useState<RoomsState | null>(null);
  const [tabHost, setTabHost] = useState<string | null>(null);
  const [destination, setDestination] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [outcome, setOutcome] = useState<SaveOutcome | null>(null);
  const [upgradeUrl, setUpgradeUrl] = useState(UPGRADE_URL);

  useEffect(() => {
    void (async () => {
      const [s, u, ls, rs, tabs] = await Promise.all([
        getSettings(),
        getUsage(),
        getLastSave(),
        getRoomsState(),
        browser.tabs.query({ active: true, currentWindow: true }),
      ]);
      setSettings(s);
      setUsage(u);
      setLastSaveState(ls);
      setRoomsState(rs);
      setDestination(s.defaultDestination);
      try {
        setTabHost(new URL(tabs[0]?.url ?? "").hostname);
      } catch {
        setTabHost(null);
      }
      setView(s.apiKey ? "main" : "connect");
      if (s.apiKey) {
        // refresh rooms/tier in the background; cheap and keeps picker fresh
        void browser.runtime.sendMessage({ type: "refresh-rooms" }).then((state) => {
          if (state) setRoomsState(state as RoomsState);
        });
      }
    })();
  }, []);

  const save = async () => {
    setSaving(true);
    setOutcome(null);
    const result = (await browser.runtime.sendMessage({
      type: "save",
      trigger: "popup",
      note: note || undefined,
      roomId: destination,
    })) as SaveOutcome;
    setSaving(false);
    setOutcome(result);
    setUsage(await getUsage());
    if (result.ok && result.lastSave) setLastSaveState(result.lastSave);
    if (result.quotaHit) {
      if (result.upgradeUrl) setUpgradeUrl(result.upgradeUrl);
      setView("quota");
    }
  };

  if (view === "loading") return <div className="popup" />;
  if (view === "connect") return <ConnectView />;
  if (view === "quota") return <QuotaView upgradeUrl={upgradeUrl} usage={usage} />;

  return (
    <div className="popup">
      <Header />
      {outcome?.firstSave ? (
        <FirstSaveCelebration topic={outcome.firstSave.topic} retrieved={outcome.firstSave.retrieved} />
      ) : (
        <>
          <DestinationPicker
            roomsState={roomsState}
            destination={destination}
            onChange={(d) => {
              setDestination(d);
              void updateSettings({ defaultDestination: d });
            }}
          />
          <textarea
            className="note"
            placeholder="Add a note (optional)"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            rows={2}
          />
          <button className="save-btn" onClick={() => void save()} disabled={saving}>
            {saving ? "Saving…" : "Save this page"}
          </button>
          <SaveStatus outcome={outcome} tabHost={tabHost} settings={settings} onSettingsChange={setSettings} />
          {lastSave && !outcome && <LastSaveRow lastSave={lastSave} />}
        </>
      )}
      <UsageMeter usage={usage} />
      <Footer tabHost={tabHost} settings={settings} onSettingsChange={setSettings} />
    </div>
  );
}

function Header() {
  return (
    <div className="header">
      <span className="logo">SenseLab</span>
      <a className="header-link" href={DASHBOARD_URL} target="_blank" rel="noreferrer">
        Dashboard ↗
      </a>
    </div>
  );
}

function ConnectView() {
  return (
    <div className="popup connect">
      <Header />
      <h2>Give your agents memory of the web</h2>
      <p className="muted">
        Save pages and highlights into SenseLab. Cursor, Claude, and every agent
        you run will remember what you read.
      </p>
      <button
        className="save-btn"
        onClick={() => void browser.tabs.create({ url: CONNECT_URL })}
      >
        Connect to SenseLab
      </button>
      <button
        className="link-btn"
        onClick={() => void browser.runtime.openOptionsPage()}
      >
        I have an API key
      </button>
    </div>
  );
}

function QuotaView({ upgradeUrl, usage }: { upgradeUrl: string; usage: UsageInfo | null }) {
  const memories = usage ? Math.floor(usage.opsLimit / OPS_PER_SAVE) : 500;
  return (
    <div className="popup connect">
      <Header />
      <h2>You’ve used your {memories} free memories this month</h2>
      <p className="muted">
        Your saved memories keep working — upgrade to keep saving. Starter gives
        you 25,000 ops/month for $29.
      </p>
      <button
        className="save-btn"
        onClick={() => {
          void track("extension_upgrade_cta_clicked", { from: "quota_screen" });
          void browser.tabs.create({ url: upgradeUrl });
        }}
      >
        Upgrade
      </button>
      <p className="tiny muted">Quota resets at the start of your next billing month.</p>
    </div>
  );
}

function FirstSaveCelebration({ topic, retrieved }: { topic: string; retrieved: boolean }) {
  const prompt = `What did I save about "${topic}"?`;
  const [copied, setCopied] = useState(false);
  return (
    <div className="first-save">
      <h2>Saved — and already retrievable</h2>
      <p className="muted">
        {retrieved
          ? "We just asked your memory about this page and it answered. Every agent you run can now do the same."
          : "This page is in your memory. Every agent you run can now recall it."}
      </p>
      <div className="prompt-box">
        <code>{prompt}</code>
        <button
          className="copy-btn"
          onClick={() => {
            void navigator.clipboard.writeText(prompt);
            setCopied(true);
          }}
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <p className="tiny muted">Paste this into Cursor or Claude to see it work.</p>
    </div>
  );
}

function DestinationPicker({
  roomsState,
  destination,
  onChange,
}: {
  roomsState: RoomsState | null;
  destination: string | null;
  onChange: (d: string | null) => void;
}) {
  const unlocked = roomsState?.roomsUnlocked ?? false;
  const rooms = roomsState?.rooms ?? [];
  return (
    <div className="destination">
      <label className="tiny muted">Save to</label>
      <div className="dest-options">
        <button
          className={`dest ${destination === null ? "active" : ""}`}
          onClick={() => onChange(null)}
        >
          My memory
        </button>
        {unlocked ? (
          rooms.length > 0 ? (
            rooms.map((r) => (
              <button
                key={r.room_id}
                className={`dest ${destination === r.room_id ? "active" : ""}`}
                onClick={() => onChange(r.room_id)}
              >
                {(r.name as string) ?? r.room_id}
              </button>
            ))
          ) : (
            <span className="tiny muted dest-empty">No rooms yet</span>
          )
        ) : (
          <button
            className="dest locked"
            onClick={() => {
              void track("extension_rooms_cta_clicked");
              void browser.tabs.create({ url: `${UPGRADE_URL}?feature=rooms` });
            }}
            title="Share clips with your team — available on Pro"
          >
            <span className="lock">🔒</span> Team room — Pro
          </button>
        )}
      </div>
    </div>
  );
}

function SaveStatus({
  outcome,
  tabHost,
  settings,
  onSettingsChange,
}: {
  outcome: SaveOutcome | null;
  tabHost: string | null;
  settings: Settings | null;
  onSettingsChange: (s: Settings) => void;
}) {
  if (!outcome) return null;
  if (outcome.ok && outcome.lastSave) {
    const updated = (outcome.lastSave.version ?? 1) > 1;
    return (
      <p className="status ok">
        {updated ? "Already saved — updated" : "Saved"}
        {outcome.lastSave.destination === "room" ? " to room" : ""} ·{" "}
        <a href={DASHBOARD_URL} target="_blank" rel="noreferrer">
          View in dashboard
        </a>
      </p>
    );
  }
  if (outcome.blocked === "blocklist") {
    return (
      <p className="status warn">
        Saving is off for {tabHost} (sensitive site). Manage the list in{" "}
        <button className="inline-link" onClick={() => void browser.runtime.openOptionsPage()}>
          settings
        </button>
        .
      </p>
    );
  }
  if (outcome.blocked === "disabled-site") {
    return (
      <p className="status warn">
        You turned off saving for {tabHost}.{" "}
        <button
          className="inline-link"
          onClick={() => {
            if (!settings || !tabHost) return;
            const next = {
              ...settings,
              disabledSites: settings.disabledSites.filter((s) => s !== tabHost),
            };
            void updateSettings({ disabledSites: next.disabledSites });
            onSettingsChange(next);
          }}
        >
          Re-enable
        </button>
      </p>
    );
  }
  if (outcome.blocked === "unsupported-page") {
    return <p className="status warn">This page can’t be saved (browser-internal page).</p>;
  }
  if (outcome.blocked === "not-connected") {
    return <p className="status err">Your API key stopped working — reconnect in settings.</p>;
  }
  return <p className="status err">{outcome.error ?? "Something went wrong."}</p>;
}

function LastSaveRow({ lastSave }: { lastSave: LastSave }) {
  return (
    <p className="tiny muted last-save">
      Last saved: <strong>{lastSave.title.slice(0, 48)}</strong>
    </p>
  );
}

function UsageMeter({ usage }: { usage: UsageInfo | null }) {
  const info = useMemo(() => {
    if (!usage) return null;
    const ratio = usage.opsLimit > 0 ? usage.opsUsed / usage.opsLimit : 0;
    const memoriesUsed = Math.floor(usage.opsUsed / OPS_PER_SAVE);
    const memoriesTotal = Math.floor(usage.opsLimit / OPS_PER_SAVE);
    return { ratio: Math.min(1, ratio), memoriesUsed, memoriesTotal };
  }, [usage]);
  if (!info) return null;

  const level =
    info.ratio >= CRITICAL_THRESHOLD ? "critical" : info.ratio >= WARN_THRESHOLD ? "warn" : "ok";
  return (
    <div className="usage">
      <div className="usage-bar">
        <div className={`usage-fill ${level}`} style={{ width: `${info.ratio * 100}%` }} />
      </div>
      <div className="usage-row">
        <span className="tiny muted">
          {info.memoriesUsed} / {info.memoriesTotal} memories this month
          {usage?.source === "local" ? " (approx.)" : ""}
        </span>
        {level !== "ok" && (
          <button
            className="inline-link tiny"
            onClick={() => {
              void track("extension_upgrade_cta_clicked", { from: "usage_meter" });
              void browser.tabs.create({ url: UPGRADE_URL });
            }}
          >
            Upgrade
          </button>
        )}
      </div>
    </div>
  );
}

function Footer({
  tabHost,
  settings,
  onSettingsChange,
}: {
  tabHost: string | null;
  settings: Settings | null;
  onSettingsChange: (s: Settings) => void;
}) {
  const disabled = !!tabHost && !!settings?.disabledSites.includes(tabHost);
  return (
    <div className="footer">
      {tabHost && settings && (
        <label className="site-toggle tiny">
          <input
            type="checkbox"
            checked={disabled}
            onChange={(e) => {
              const disabledSites = e.target.checked
                ? [...settings.disabledSites, tabHost]
                : settings.disabledSites.filter((s) => s !== tabHost);
              void updateSettings({ disabledSites });
              onSettingsChange({ ...settings, disabledSites });
            }}
          />
          Never save on {tabHost}
        </label>
      )}
      <button className="inline-link tiny" onClick={() => void browser.runtime.openOptionsPage()}>
        Settings
      </button>
    </div>
  );
}
