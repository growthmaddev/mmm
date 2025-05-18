import { Switch, Route } from "wouter";
import { TooltipProvider } from "./components/ui/tooltip";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { useAuth } from "./hooks/useAuth";

// Import pages
import HomePage from "./pages/home";
import Dashboard from "./pages/dashboard";
import Projects from "./pages/projects";
import CreateProject from "./pages/projects/create";
import ProjectDetails from "./pages/projects/[id]/index";
import ProjectDataUpload from "./pages/projects/[id]/data-upload";
import ProjectModelSetup from "./pages/projects/[id]/model-setup";
import ProjectResults from "./pages/projects/[id]/results";
import ProjectBudgetOptimizer from "./pages/projects/[id]/budget-optimizer";
import NotFound from "./pages/not-found";

function App() {
  const { isAuthenticated, isLoading } = useAuth();

  // Show home page to non-authenticated users
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Switch>
          {/* Public route */}
          <Route path="/" component={isAuthenticated ? Dashboard : HomePage} />
          
          {/* Protected routes */}
          <Route path="/dashboard" component={Dashboard} />
          <Route path="/projects" component={Projects} />
          <Route path="/projects/create" component={CreateProject} />
          <Route path="/projects/:id" component={ProjectDetails} />
          <Route path="/projects/:id/data-upload" component={ProjectDataUpload} />
          <Route path="/projects/:id/model-setup" component={ProjectModelSetup} />
          <Route path="/projects/:id/results" component={ProjectResults} />
          <Route path="/projects/:id/budget-optimizer" component={ProjectBudgetOptimizer} />
          
          {/* 404 Fallback */}
          <Route component={NotFound} />
        </Switch>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
