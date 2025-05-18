import { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface StatCardProps {
  title: string;
  value: string | number;
  icon: ReactNode;
  color?: "primary" | "secondary" | "accent";
}

export default function StatCard({
  title,
  value,
  icon,
  color = "primary",
}: StatCardProps) {
  const getColorClasses = (color: string) => {
    switch (color) {
      case "primary":
        return "bg-primary-100 text-primary-600";
      case "secondary":
        return "bg-secondary-100 text-secondary-600";
      case "accent":
        return "bg-orange-100 text-orange-600";
      default:
        return "bg-primary-100 text-primary-600";
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-slate-200">
      <div className="flex items-center">
        <div className={cn("p-3 rounded-full", getColorClasses(color))}>
          {icon}
        </div>
        <div className="ml-4">
          <h3 className="text-sm font-medium text-slate-500">{title}</h3>
          <p className="text-2xl font-semibold">{value}</p>
        </div>
      </div>
    </div>
  );
}
