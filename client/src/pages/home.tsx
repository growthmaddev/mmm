import { useAuth } from "../hooks/useAuth";
import { Link } from "wouter";
import { Button } from "../components/ui/button";

export default function HomePage() {
  const { isLoading } = useAuth();

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      <div className="container max-w-7xl mx-auto px-4 py-12">
        {/* Header/Navigation */}
        <header className="flex justify-between items-center py-6">
          <div className="flex items-center space-x-2">
            <h1 className="text-2xl font-bold text-primary">MMM Platform</h1>
          </div>
          <div>
            <Button 
              variant="outline" 
              onClick={() => window.location.href = "/api/login"}
              disabled={isLoading}
            >
              {isLoading ? "Loading..." : "Sign In"}
            </Button>
          </div>
        </header>

        {/* Hero Section */}
        <section className="py-20 text-center">
          <h1 className="text-5xl font-bold text-slate-900 mb-6">
            Market Mix Modelling <span className="text-primary">Made Simple</span>
          </h1>
          <p className="text-xl text-slate-600 max-w-3xl mx-auto mb-10">
            Understand your marketing effectiveness without technical expertise. Our platform helps marketers analyze channel performance and optimize budget allocation.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Button 
              size="lg"
              onClick={() => window.location.href = "/api/login"}
              disabled={isLoading}
              className="px-8 py-6 text-lg"
            >
              {isLoading ? "Loading..." : "Get Started"}
            </Button>
            <Button 
              variant="outline" 
              size="lg"
              asChild
              className="px-8 py-6 text-lg"
            >
              <Link href="/about">Learn More</Link>
            </Button>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-20">
          <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white p-8 rounded-xl shadow-sm border border-slate-200">
              <div className="w-12 h-12 flex items-center justify-center bg-primary/10 text-primary rounded-lg mb-6">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-upload"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>
              </div>
              <h3 className="text-xl font-semibold mb-4">1. Upload Your Data</h3>
              <p className="text-slate-600">
                Simply upload your marketing data from any source or connect to your existing analytics platforms.
              </p>
            </div>
            <div className="bg-white p-8 rounded-xl shadow-sm border border-slate-200">
              <div className="w-12 h-12 flex items-center justify-center bg-primary/10 text-primary rounded-lg mb-6">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-settings"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
              </div>
              <h3 className="text-xl font-semibold mb-4">2. Configure Your Model</h3>
              <p className="text-slate-600">
                Use our simple interface to set up your model parameters - no coding or statistical knowledge required.
              </p>
            </div>
            <div className="bg-white p-8 rounded-xl shadow-sm border border-slate-200">
              <div className="w-12 h-12 flex items-center justify-center bg-primary/10 text-primary rounded-lg mb-6">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-bar-chart-3"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
              </div>
              <h3 className="text-xl font-semibold mb-4">3. Get Actionable Insights</h3>
              <p className="text-slate-600">
                View clear visualizations of your marketing performance and get AI-powered recommendations for budget allocation.
              </p>
            </div>
          </div>
        </section>

        {/* Call to Action */}
        <section className="py-20">
          <div className="bg-primary/5 border border-primary/20 rounded-2xl p-12 text-center">
            <h2 className="text-3xl font-bold mb-6">Ready to Optimize Your Marketing Mix?</h2>
            <p className="text-xl text-slate-600 max-w-3xl mx-auto mb-8">
              Join other marketing teams who have improved their ROI by up to 30% using our platform.
            </p>
            <Button 
              size="lg"
              onClick={() => window.location.href = "/api/login"}
              disabled={isLoading}
              className="px-8 py-6 text-lg"
            >
              {isLoading ? "Loading..." : "Get Started for Free"}
            </Button>
          </div>
        </section>

        {/* Footer */}
        <footer className="py-10 border-t border-slate-200 mt-10">
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-xl font-bold text-primary mb-4">MMM Platform</h3>
              <p className="text-slate-500">
                Making Market Mix Modelling accessible for all marketing teams.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Resources</h4>
              <ul className="space-y-2">
                <li><Link href="/blog"><a className="text-slate-500 hover:text-primary">Blog</a></Link></li>
                <li><Link href="/documentation"><a className="text-slate-500 hover:text-primary">Documentation</a></Link></li>
                <li><Link href="/support"><a className="text-slate-500 hover:text-primary">Support</a></Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Legal</h4>
              <ul className="space-y-2">
                <li><Link href="/terms"><a className="text-slate-500 hover:text-primary">Terms of Service</a></Link></li>
                <li><Link href="/privacy"><a className="text-slate-500 hover:text-primary">Privacy Policy</a></Link></li>
              </ul>
            </div>
          </div>
          <div className="text-center text-slate-400 mt-10">
            &copy; {new Date().getFullYear()} MMM Platform. All rights reserved.
          </div>
        </footer>
      </div>
    </div>
  );
}