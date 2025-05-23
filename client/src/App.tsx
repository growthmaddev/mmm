import { Switch, Route, Router, useLocation } from "wouter";
import { TooltipProvider } from "./components/ui/tooltip";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";

// Import pages
import HomePage from "./pages/home";
import LoginPage from "./pages/auth/login";
import RegisterPage from "./pages/auth/register";
import Dashboard from "./pages/dashboard";
import OptimizationPage from "./pages/dashboard/OptimizationPage";
import Projects from "./pages/projects";
import CreateProject from "./pages/projects/create";
import ProjectDetails from "./pages/projects/[id]/index";
import ProjectDataUpload from "./pages/projects/[id]/data-upload";
import ColumnMapping from "./pages/projects/[id]/column-mapping-direct";
import ProjectModelSetup from "./pages/projects/[id]/model-setup";
import ProjectResults from "./pages/projects/[id]/results";
import ProjectBudgetOptimizer from "./pages/projects/[id]/budget-optimizer";
import NotFound from "./pages/not-found";

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Switch>
          {/* Public routes */}
          <Route path="/" component={HomePage} />
          <Route path="/login" component={LoginPage} />
          <Route path="/register" component={RegisterPage} />
          
          {/* Dashboard routes */}
          <Route path="/dashboard" component={Dashboard} />
          <Route path="/dashboard/optimization" component={OptimizationPage} />
          
          {/* Projects routes */}
          <Route path="/projects" component={Projects} />
          <Route path="/projects/create" component={CreateProject} />
          <Route path="/projects/:id" component={ProjectDetails} />
          <Route path="/projects/:id/data-upload" component={ProjectDataUpload} />
          <Route path="/projects/:id/column-mapping-direct" component={ColumnMapping} />
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
