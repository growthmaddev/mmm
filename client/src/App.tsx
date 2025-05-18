import { Switch, Route } from "wouter";
import { TooltipProvider } from "@/components/ui/tooltip";

// Import pages
import Dashboard from "@/pages/dashboard";
import Login from "@/pages/auth/login";
import Register from "@/pages/auth/register";
import Projects from "@/pages/projects";
import CreateProject from "@/pages/projects/create";
import ProjectDetails from "@/pages/projects/[id]/index";
import ProjectDataUpload from "@/pages/projects/[id]/data-upload";
import ProjectModelSetup from "@/pages/projects/[id]/model-setup";
import ProjectResults from "@/pages/projects/[id]/results";
import ProjectBudgetOptimizer from "@/pages/projects/[id]/budget-optimizer";
import NotFound from "@/pages/not-found";

function App() {
  return (
    <TooltipProvider>
      <Switch>
        {/* Auth Routes */}
        <Route path="/login" component={Login} />
        <Route path="/register" component={Register} />
        
        {/* Dashboard */}
        <Route path="/" component={Dashboard} />
        
        {/* Projects */}
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
  );
}

export default App;
