import { useEffect, useState } from "react";
import { whoami } from "@/utils/api";
import { DEFAULT_BLOCKLIST } from "@/utils/blocklist";
import { CONNECT_URL } from "@/utils/config";
import { getSettings, updateSettings } from "@/utils/storage";
import type { Settings } from "@/utils/types";

export default function Options() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [keyInput, setKeyInput] = useState("");
  const [keyStatus, setKeyStatus] = useState<"idle" | "checking" | "ok" | "bad">("idle");
  const [blocklistInput, setBlocklistInput] = useState("");

  useEffect(() => {
    void getSettings().then((s) => {
      setSettings(s);
      setBlocklistInput(s.customBlocklist.join("\n"));
    });
  }, []);

  if (!settings) return null;

  const patch = async (p: Partial<Settings>) => {
    const next = await updateSettings(p);
    setSettings(next);
  };

  const saveKey = async () => {
    const key = keyInput.trim();
    if (!key) return;
    setKeyStatus("checking");
    try {
      await whoami(key);
      await patch({ apiKey: key });
      setKeyStatus("ok");
      setKeyInput("");
    } catch {
      setKeyStatus("bad");
    }
  };

  return (
    <div className="options">
      <h1>SenseLab — Save to Memory</h1>

      <section>
        <h2>Account</h2>
        {settings.apiKey ? (
          <div className="row">
            <p>
              Connected{settings.accountEmail ? ` as ${settings.accountEmail}` : ""} (key ending{" "}
              …{settings.apiKey.slice(-4)})
            </p>
            <button
              className="danger"
              onClick={() => void patch({ apiKey: null, accountEmail: null, firstSaveDone: false })}
            >
              Disconnect
            </button>
          </div>
        ) : (
          <p>
            Not connected. <a href={CONNECT_URL}>Connect via the dashboard</a> or paste an API key
            below.
          </p>
        )}
        <div className="row">
          <input
            type="password"
            placeholder="amfs_…"
            value={keyInput}
            onChange={(e) => {
              setKeyInput(e.target.value);
              setKeyStatus("idle");
            }}
          />
          <button onClick={() => void saveKey()} disabled={keyStatus === "checking" || !keyInput.trim()}>
            {keyStatus === "checking" ? "Checking…" : "Save key"}
          </button>
        </div>
        {keyStatus === "ok" && <p className="ok">Key verified and saved.</p>}
        {keyStatus === "bad" && <p className="err">That key didn’t work — check it in your dashboard (Settings → API Keys).</p>}
      </section>

      <section>
        <h2>What gets sent when you save</h2>
        <p className="muted">
          Only when you explicitly save (button, shortcut, or right-click), the extension sends:
          the page URL, its title, the readable article text (or your selection), and your
          optional note. Nothing is ever collected in the background while you browse.
        </p>
      </section>

      <section>
        <h2>Sites where saving is disabled</h2>
        <p className="muted">
          Built-in protections cover banking, health, webmail, and password managers (
          {DEFAULT_BLOCKLIST.length} patterns). Add your own below, one per line — use{" "}
          <code>*.example.com</code> to match subdomains.
        </p>
        <textarea
          rows={5}
          value={blocklistInput}
          onChange={(e) => setBlocklistInput(e.target.value)}
          onBlur={() =>
            void patch({
              customBlocklist: blocklistInput
                .split("\n")
                .map((l) => l.trim())
                .filter(Boolean),
            })
          }
          placeholder="*.internal-wiki.mycompany.com"
        />
        {settings.disabledSites.length > 0 && (
          <>
            <h3>Per-site “never save here” toggles</h3>
            <ul className="disabled-sites">
              {settings.disabledSites.map((site) => (
                <li key={site}>
                  {site}{" "}
                  <button
                    className="inline"
                    onClick={() =>
                      void patch({ disabledSites: settings.disabledSites.filter((s) => s !== site) })
                    }
                  >
                    re-enable
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
      </section>

      <section>
        <h2>Privacy</h2>
        <label className="check">
          <input
            type="checkbox"
            checked={settings.analyticsEnabled}
            onChange={(e) => void patch({ analyticsEnabled: e.target.checked })}
          />
          Share anonymous usage analytics (helps us improve the extension)
        </label>
      </section>
    </div>
  );
}
