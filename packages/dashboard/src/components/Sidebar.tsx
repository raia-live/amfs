"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { clsx } from "clsx";
import {
  GitBranch,
  GitPullRequest,
  Tag,
  Clock,
  RotateCcw,
  LayoutDashboard,
} from "lucide-react";

const NAV = [
  { href: "/", label: "Timeline", icon: Clock },
  { href: "/branches", label: "Branches", icon: GitBranch },
  { href: "/pull-requests", label: "Pull Requests", icon: GitPullRequest },
  { href: "/tags", label: "Tags", icon: Tag },
  { href: "/rollback", label: "Rollback", icon: RotateCcw },
] as const;

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-56 bg-void-dark border-r border-void-light flex flex-col z-40">
      <Link
        href="/"
        className="flex items-center gap-2 px-5 py-5 border-b border-void-light"
      >
        <LayoutDashboard className="w-5 h-5 text-sacred-gold" />
        <span className="text-lg font-semibold tracking-tight text-sacred-gold">
          AMFS
        </span>
        <span className="text-xs bg-sacred-gold/20 text-sacred-gold px-1.5 py-0.5 rounded font-medium ml-auto">
          PRO
        </span>
      </Link>

      <nav className="flex-1 py-3 px-2 space-y-0.5">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active =
            href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all",
                active
                  ? "bg-sacred-gold/15 text-sacred-gold font-medium"
                  : "text-text-secondary hover:bg-void-light hover:text-text-primary",
              )}
            >
              <Icon className="w-4 h-4" />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="px-4 py-4 border-t border-void-light">
        <div className="text-xs text-text-muted">Agent Memory as Git</div>
        <div className="text-xs text-text-muted mt-0.5">Sacred Timeline</div>
      </div>
    </aside>
  );
}
