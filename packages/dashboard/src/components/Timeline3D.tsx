"use client";

import { useRef, useMemo, useState, useCallback } from "react";
import { Canvas, useFrame, ThreeEvent } from "@react-three/fiber";
import { OrbitControls, Text, Line, Billboard, Float } from "@react-three/drei";
import * as THREE from "three";
import type { TimelineEvent } from "@/lib/api";

const EVENT_COLORS: Record<string, string> = {
  write: "#c8a84e",
  outcome: "#22c55e",
  webhook: "#4a9eff",
  brief_compiled: "#a78bfa",
  cross_agent_read: "#f472b6",
  branch_created: "#fb923c",
  branch_merged: "#4ade80",
  branch_closed: "#ef4444",
  access_granted: "#06b6d4",
  access_revoked: "#f87171",
  rollback: "#eab308",
  tag_created: "#818cf8",
  cherry_pick: "#fbbf24",
  fork: "#e879f9",
};

const BRANCH_OFFSETS: Record<string, number> = {};
let nextBranchOffset = 0;

function getBranchOffset(branch: string): number {
  if (branch === "main") return 0;
  if (!(branch in BRANCH_OFFSETS)) {
    nextBranchOffset += 1;
    BRANCH_OFFSETS[branch] = nextBranchOffset;
  }
  return BRANCH_OFFSETS[branch];
}

interface EventNodeProps {
  event: TimelineEvent;
  position: [number, number, number];
  color: string;
  onSelect: (event: TimelineEvent) => void;
  selected: boolean;
}

function EventNode({ event, position, color, onSelect, selected }: EventNodeProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame((_, delta) => {
    if (!meshRef.current) return;
    const target = hovered || selected ? 1.4 : 1.0;
    meshRef.current.scale.lerp(
      new THREE.Vector3(target, target, target),
      delta * 8,
    );
  });

  const handleClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      onSelect(event);
    },
    [event, onSelect],
  );

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onClick={handleClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[0.15, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={hovered || selected ? 0.8 : 0.3}
          toneMapped={false}
        />
      </mesh>

      {(hovered || selected) && (
        <Billboard>
          <Text
            position={[0, 0.35, 0]}
            fontSize={0.12}
            color="#e8e6e3"
            anchorX="center"
            anchorY="bottom"
            maxWidth={3}
          >
            {event.summary?.slice(0, 60) || event.event_type}
          </Text>
          <Text
            position={[0, 0.2, 0]}
            fontSize={0.08}
            color="#9ca3af"
            anchorX="center"
            anchorY="bottom"
          >
            {event.branch} · {new Date(event.created_at).toLocaleTimeString()}
          </Text>
        </Billboard>
      )}

      <pointLight color={color} intensity={0.5} distance={2} />
    </group>
  );
}

function TimelineLine({
  points,
  color,
}: {
  points: [number, number, number][];
  color: string;
}) {
  if (points.length < 2) return null;
  return (
    <Line
      points={points}
      color={color}
      lineWidth={2}
      transparent
      opacity={0.6}
    />
  );
}

function BranchLabel({ branch, position }: { branch: string; position: [number, number, number] }) {
  return (
    <Billboard position={position}>
      <Float speed={1} rotationIntensity={0} floatIntensity={0.3}>
        <Text
          fontSize={0.18}
          color={branch === "main" ? "#c8a84e" : "#4a9eff"}
          anchorX="center"
          outlineWidth={0.01}
          outlineColor="#0a0a0f"
        >
          {branch}
        </Text>
      </Float>
    </Billboard>
  );
}

function Scene({
  events,
  selectedEvent,
  onSelect,
}: {
  events: TimelineEvent[];
  selectedEvent: TimelineEvent | null;
  onSelect: (event: TimelineEvent) => void;
}) {
  const sortedEvents = useMemo(
    () =>
      [...events].sort(
        (a, b) =>
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      ),
    [events],
  );

  const { nodes, lines, labels } = useMemo(() => {
    const branchPoints: Record<string, [number, number, number][]> = {};
    const nodeList: {
      event: TimelineEvent;
      position: [number, number, number];
      color: string;
    }[] = [];

    const branchMap: Record<string, number> = {};
    let bIdx = 0;

    sortedEvents.forEach((ev, i) => {
      const branch = ev.branch || "main";
      if (!(branch in branchMap)) {
        branchMap[branch] = branch === "main" ? 0 : ++bIdx;
      }
      const x = i * 1.2;
      const y = branchMap[branch] * 2.5;
      const z = 0;
      const pos: [number, number, number] = [x, y, z];

      if (!branchPoints[branch]) branchPoints[branch] = [];
      branchPoints[branch].push(pos);

      nodeList.push({
        event: ev,
        position: pos,
        color: EVENT_COLORS[ev.event_type] || "#6b7280",
      });
    });

    const lineList = Object.entries(branchPoints).map(([branch, pts]) => ({
      branch,
      points: pts,
      color: branch === "main" ? "#c8a84e" : "#4a9eff",
    }));

    const labelList = Object.entries(branchPoints).map(([branch, pts]) => {
      const first = pts[0];
      return {
        branch,
        position: [first[0] - 0.8, first[1], first[2]] as [number, number, number],
      };
    });

    return { nodes: nodeList, lines: lineList, labels: labelList };
  }, [sortedEvents]);

  return (
    <>
      <ambientLight intensity={0.15} />
      <directionalLight position={[10, 10, 5]} intensity={0.3} />
      <fog attach="fog" args={["#0a0a0f", 15, 60]} />

      {lines.map((l) => (
        <TimelineLine
          key={l.branch}
          points={l.points}
          color={l.color}
        />
      ))}

      {labels.map((l) => (
        <BranchLabel key={l.branch} branch={l.branch} position={l.position} />
      ))}

      {nodes.map((n) => (
        <EventNode
          key={n.event.id}
          event={n.event}
          position={n.position}
          color={n.color}
          onSelect={onSelect}
          selected={selectedEvent?.id === n.event.id}
        />
      ))}

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.05}
        minDistance={3}
        maxDistance={80}
      />
    </>
  );
}

interface Timeline3DProps {
  events: TimelineEvent[];
}

export function Timeline3D({ events }: Timeline3DProps) {
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(
    null,
  );

  return (
    <div className="relative w-full h-full min-h-[600px]">
      <Canvas
        camera={{ position: [5, 3, 10], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <Scene
          events={events}
          selectedEvent={selectedEvent}
          onSelect={setSelectedEvent}
        />
      </Canvas>

      {selectedEvent && (
        <div className="absolute bottom-4 left-4 right-4 max-w-lg bg-void-dark/95 backdrop-blur-md border border-void-light rounded-xl p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span
              className="inline-block px-2 py-0.5 rounded text-xs font-medium"
              style={{
                backgroundColor:
                  (EVENT_COLORS[selectedEvent.event_type] || "#6b7280") + "20",
                color: EVENT_COLORS[selectedEvent.event_type] || "#6b7280",
              }}
            >
              {selectedEvent.event_type}
            </span>
            <span className="text-xs text-text-muted">
              {new Date(selectedEvent.created_at).toLocaleString()}
            </span>
          </div>
          <p className="text-sm text-text-primary mb-1">
            {selectedEvent.summary}
          </p>
          <div className="flex items-center gap-3 text-xs text-text-secondary">
            <span>Branch: {selectedEvent.branch}</span>
            <span>Agent: {selectedEvent.agent_id}</span>
          </div>
          {selectedEvent.details &&
            Object.keys(selectedEvent.details).length > 0 && (
              <pre className="mt-2 text-xs text-text-muted bg-void-black/50 rounded p-2 max-h-32 overflow-auto">
                {JSON.stringify(selectedEvent.details, null, 2)}
              </pre>
            )}
          <button
            onClick={() => setSelectedEvent(null)}
            className="absolute top-2 right-2 text-text-muted hover:text-text-primary text-xs"
          >
            ✕
          </button>
        </div>
      )}

      <div className="absolute top-4 right-4 flex flex-wrap gap-2">
        {Object.entries(EVENT_COLORS)
          .slice(0, 8)
          .map(([type, color]) => (
            <div key={type} className="flex items-center gap-1 text-xs text-text-muted">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: color }}
              />
              {type.replace(/_/g, " ")}
            </div>
          ))}
      </div>
    </div>
  );
}
