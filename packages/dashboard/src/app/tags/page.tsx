"use client";

import { useState, useEffect } from "react";
import { Tag as TagIcon, Plus, Trash2, X } from "lucide-react";
import { api, Tag } from "@/lib/api";

export default function TagsPage() {
  const [tags, setTags] = useState<Tag[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);

  async function fetchTags() {
    setLoading(true);
    try {
      const data = await api.listTags();
      setTags(data.tags || []);
    } catch {
      setTags([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchTags();
  }, []);

  async function handleCreate(name: string, branch: string, desc: string) {
    await api.createTag(name, branch, desc || undefined);
    setShowCreate(false);
    fetchTags();
  }

  async function handleDelete(name: string) {
    if (!confirm(`Delete tag "${name}"?`)) return;
    await api.deleteTag(name);
    fetchTags();
  }

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <TagIcon className="w-5 h-5 text-branch-purple" />
          <h1 className="text-xl font-semibold">Tags / Snapshots</h1>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-1.5 px-3 py-2 bg-branch-purple/20 text-branch-purple border border-branch-purple/30 rounded-lg text-sm hover:bg-branch-purple/30 transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Tag
        </button>
      </div>

      {showCreate && (
        <CreateTagModal
          onClose={() => setShowCreate(false)}
          onCreate={handleCreate}
        />
      )}

      {loading ? (
        <div className="text-center py-12 text-text-muted">Loading…</div>
      ) : tags.length === 0 ? (
        <div className="text-center py-12 text-text-muted">
          No tags yet. Tags mark a point-in-time snapshot of memory.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {tags.map((tag) => (
            <div
              key={tag.id || tag.name}
              className="bg-void-dark border border-void-light rounded-xl p-4"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <TagIcon className="w-4 h-4 text-branch-purple" />
                  <span className="font-medium">{tag.name}</span>
                </div>
                <button
                  onClick={() => handleDelete(tag.name)}
                  className="p-1 rounded-lg hover:bg-void-light text-text-muted hover:text-danger transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
              <p className="text-xs text-text-secondary">
                Branch: <span className="text-sacred-gold">{tag.branch}</span>
              </p>
              {tag.description && (
                <p className="text-xs text-text-muted mt-1">{tag.description}</p>
              )}
              <p className="text-xs text-text-muted mt-1">
                {new Date(tag.tagged_at).toLocaleString()}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CreateTagModal({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (name: string, branch: string, desc: string) => void;
}) {
  const [name, setName] = useState("");
  const [branch, setBranch] = useState("main");
  const [desc, setDesc] = useState("");

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-void-dark border border-void-light rounded-xl p-6 w-full max-w-md">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Create Tag</h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="space-y-3">
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Tag name (e.g. v1.0, pre-migration)"
            className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-branch-purple/50"
          />
          <input
            type="text"
            value={branch}
            onChange={(e) => setBranch(e.target.value)}
            placeholder="Branch"
            className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-branch-purple/50"
          />
          <textarea
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
            placeholder="Description (optional)"
            rows={2}
            className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-branch-purple/50 resize-none"
          />
          <button
            onClick={() => onCreate(name, branch, desc)}
            disabled={!name}
            className="w-full py-2 bg-branch-purple/20 text-branch-purple border border-branch-purple/30 rounded-lg text-sm font-medium hover:bg-branch-purple/30 disabled:opacity-40 transition-colors"
          >
            Create Tag
          </button>
        </div>
      </div>
    </div>
  );
}
