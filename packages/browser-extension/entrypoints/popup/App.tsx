import { type KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { browser } from "wxt/browser";
import { track } from "@/utils/analytics";
import {
  CONNECT_URL,
  CRITICAL_THRESHOLD,
  DASHBOARD_URL,
  OPS_PER_SAVE,
  ROOMS_URL,
  UPGRADE_URL,
  WARN_THRESHOLD,
} from "@/utils/config";
import {
  type Destination,
  buildDestinations,
  destinationIndex,
  filterDestinations,
  highlightRuns,
  queryTokens,
} from "@/utils/destinations";
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
      <img className="logo" src="/senselab-logo-dark.png" alt="SenseLab" />
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
        onClick={() =>
          void browser.tabs.create({
            // ext id lets the dashboard validate the target and address
            // chrome.runtime.sendMessage back to this extension.
            url: `${CONNECT_URL}?ext=${browser.runtime.id}`,
          })
        }
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

/**
 * Single-select combobox: collapsed it shows only the current destination, so
 * an account with dozens of rooms costs one line instead of a wall of chips.
 * Open, it filters by room name or memory path as you type.
 */
function DestinationPicker({
  roomsState,
  destination,
  onChange,
}: {
  roomsState: RoomsState | null;
  destination: string | null;
  onChange: (d: string | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const unlocked = roomsState?.roomsUnlocked ?? false;
  const destinations = useMemo(() => buildDestinations(roomsState), [roomsState]);
  const tokens = useMemo(() => queryTokens(query), [query]);
  const results = useMemo(() => filterDestinations(destinations, query), [destinations, query]);
  // An unknown id means the room was deleted or the tier was downgraded — the
  // background falls back to personal memory, so show that rather than a UUID.
  const selected = destinations[destinationIndex(destinations, destination)];

  useEffect(() => {
    if (!open) {
      setQuery("");
      return;
    }
    inputRef.current?.focus();
    const onPointerDown = (e: MouseEvent) => {
      if (!boxRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [open]);

  useEffect(() => setCursor(0), [query]);

  useEffect(() => {
    if (!open) return;
    listRef.current
      ?.querySelector(`[data-idx="${cursor}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [open, cursor, results.length]);

  // Open on the current destination so a room far down a long list is both
  // highlighted and scrolled into view.
  const openPicker = () => {
    setCursor(destinationIndex(destinations, destination));
    setOpen(true);
  };

  const commit = (d: Destination | undefined) => {
    if (!d) return;
    onChange(d.id);
    setOpen(false);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setCursor((c) => Math.min(c + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setCursor((c) => Math.max(c - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      commit(results[cursor]);
    } else if (e.key === "Escape" || e.key === "Tab") {
      setOpen(false);
    }
  };

  return (
    <div className="destination" ref={boxRef}>
      <span className="tiny muted dest-label" id="dest-label">
        Save to
      </span>
      {open ? (
        <>
          <input
            ref={inputRef}
            className="dest-search"
            role="combobox"
            aria-labelledby="dest-label"
            aria-expanded
            aria-controls="dest-list"
            aria-autocomplete="list"
            aria-activedescendant={results[cursor] ? `dest-opt-${cursor}` : undefined}
            placeholder="Search rooms or memory paths…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
          />
          <div className="dest-menu">
            <div className="dest-list" id="dest-list" role="listbox" ref={listRef}>
              {results.length === 0 && (
                <p className="dest-none tiny muted">No destination matches “{query}”</p>
              )}
              {results.map((d, i) => (
                <button
                  key={d.id ?? "personal"}
                  id={`dest-opt-${i}`}
                  data-idx={i}
                  role="option"
                  aria-selected={d.id === selected.id}
                  className={`dest-item ${i === cursor ? "cursor" : ""} ${
                    d.id === selected.id ? "selected" : ""
                  }`}
                  onMouseMove={() => setCursor(i)}
                  onClick={() => commit(d)}
                >
                  <span className="dest-item-name">
                    {/* Highlight runs live in their own inline span: the flex
                        gap on the row would otherwise space out every run. */}
                    <span className="dest-item-text">
                      <Highlighted text={d.label} tokens={tokens} />
                    </span>
                    {d.id === selected.id && <span className="dest-check">✓</span>}
                  </span>
                  {d.hint && (
                    <span className="dest-item-hint">
                      <Highlighted text={d.hint} tokens={tokens} />
                    </span>
                  )}
                </button>
              ))}
            </div>
            {!unlocked ? (
              <button
                className="dest-upsell"
                onClick={() => {
                  void track("extension_rooms_cta_clicked");
                  void browser.tabs.create({ url: ROOMS_URL });
                }}
              >
                <span className="lock">🔒</span> Save into a shared team room — Pro
              </button>
            ) : (
              destinations.length === 1 && (
                <p className="dest-foot tiny muted">
                  No rooms yet —{" "}
                  <a href={ROOMS_URL} target="_blank" rel="noreferrer">
                    create one
                  </a>
                  .
                </p>
              )
            )}
          </div>
        </>
      ) : (
        <button
          className="dest-trigger"
          aria-haspopup="listbox"
          aria-expanded={false}
          aria-labelledby="dest-label dest-value"
          onClick={openPicker}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              openPicker();
            }
          }}
        >
          <span className="dest-trigger-label" id="dest-value">
            {selected.label}
          </span>
          <span className="chev">▾</span>
        </button>
      )}
    </div>
  );
}

function Highlighted({ text, tokens }: { text: string; tokens: string[] }) {
  if (tokens.length === 0) return <>{text}</>;
  return (
    <>
      {highlightRuns(text, tokens).map((run, i) =>
        run.match ? <mark key={i}>{run.text}</mark> : <span key={i}>{run.text}</span>,
      )}
    </>
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
