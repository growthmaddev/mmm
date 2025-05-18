import * as React from "react";
import { cn } from "@/lib/utils";
import { CheckIcon } from "lucide-react";

export interface StepsProps extends React.HTMLAttributes<HTMLDivElement> {
  currentStep: number;
}

export function Steps({ currentStep, className, ...props }: StepsProps) {
  const children = React.Children.toArray(props.children);
  
  return (
    <div className={cn("flex items-center", className)} {...props}>
      {children.map((step, index) => React.cloneElement(step as React.ReactElement, {
        index,
        isActive: currentStep === index,
        isCompleted: currentStep > index,
        isLast: index === children.length - 1,
      }))}
    </div>
  );
}

export interface StepProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string;
  description?: string;
  index?: number;
  isActive?: boolean;
  isCompleted?: boolean;
  isLast?: boolean;
  onClick?: () => void;
}

export function Step({
  title,
  description,
  index,
  isActive,
  isCompleted,
  isLast,
  onClick,
  className,
  ...props
}: StepProps) {
  return (
    <div
      className={cn(
        "flex flex-1 flex-col",
        isLast ? "" : "border-r border-slate-200 dark:border-slate-700",
        className
      )}
      {...props}
    >
      <div className="flex items-center space-x-2 px-4">
        <div
          onClick={onClick}
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium border transition-colors",
            isActive ?
              "border-primary bg-primary text-primary-foreground" :
              isCompleted ?
                "border-primary bg-primary text-primary-foreground cursor-pointer" :
                "border-slate-300 dark:border-slate-600 cursor-pointer",
            onClick && "cursor-pointer"
          )}
        >
          {isCompleted ? (
            <CheckIcon className="h-4 w-4" />
          ) : (
            <span>{(index || 0) + 1}</span>
          )}
        </div>
        <div className="flex flex-col">
          <div
            className={cn(
              "text-sm font-medium",
              isActive || isCompleted ?
                "text-slate-900 dark:text-slate-100" :
                "text-slate-500 dark:text-slate-400"
            )}
          >
            {title}
          </div>
          {description && (
            <div className="text-xs text-slate-500 dark:text-slate-400">
              {description}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
