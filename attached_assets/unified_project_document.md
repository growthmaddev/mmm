# Unified Project Documentation

## Project Requirements Document (PRD)

### 1. Project Overview

We are building a web-based Market Mix Modelling (MMM) platform that helps non-technical marketers, strategists, and agency teams understand how their marketing spend drives their business outcomes. By guiding users through data upload or connector setup, question-driven model configuration, and clear result visualization, the platform turns complex Bayesian MMM into a straightforward workflow.

This MVP aims to cover the end-to-end journey: secure data input with validation, transparent model training powered by PyMC-Marketing, interactive dashboards showing channel contributions and saturation curves, AI-generated plain-English summaries, and a budget optimization tool. We’ll measure success by how easily a non-expert can run a full MMM analysis, understand the insights, and apply recommendations to optimize their marketing budget.

### 2. In-Scope vs. Out-of-Scope

**In-Scope (MVP v1.0)**

*   User sign-up and login via email/password, with multi-tenant workspaces per organization
*   Super-admin console for onboarding organizations, user management, and basic platform health metrics
*   CSV/Excel file uploads (with downloadable templates) and OAuth connectors for Google Ads, Facebook Ads, and Google Analytics
*   Automated data validation for missing values, date formats, and types, plus user guidance on imperfect data
*   Question-driven Bayesian MMM setup using PyMC-Marketing (modeling adstock and saturation)
*   Progress indicators and status updates during model training on a Replit backend
*   Interactive results dashboard with ROI metrics, contribution charts, response curves, and hover-over tooltips
*   AI-generated natural-language summaries, PDF/CSV exports, and budget optimization scenario builder with live recommendations
*   Email notifications and webhooks for key events (model completion, errors)
*   Audit logs capturing uploads, model runs, and user actions for admin review

**Out-of-Scope (Post-MVP)**

*   Single sign-on (SSO) and two-factor authentication
*   Advanced MMM engines beyond PyMC-Marketing (e.g., RobynPy, Google Meridian)
*   Granular roles and permissions inside each organization
*   Model version control or lift test calibration
*   Custom domains and SSL certificate management (planned later)
*   Connectors beyond the three initial APIs

### 3. User Flow

A marketer arrives at the platform’s homepage and creates an account by entering their email and a password. After logging in, they land on a dashboard listing all workspaces for their organization. They can create a new project workspace or open an existing one. Clear labels and security badges reinforce data privacy between organizations.

Inside a workspace, users choose between uploading their CSV/Excel files or connecting via APIs to Google Ads, Facebook Ads, or Google Analytics. The platform guides them to map columns or authenticate with OAuth, then runs automated checks to catch missing data or formatting issues. Once data is validated, the user answers high-level business questions that the system maps to an MMM configuration. Model training runs in the background with a visible progress bar and status updates. When complete, users are taken to an interactive insights dashboard where they can view charts, read an AI-written summary, download reports, or open the budget optimizer to test spending scenarios.

### 4. Core Features

*   Authentication & Multi-Tenancy: email/password login, isolated workspaces, super-admin console
*   Data Management: CSV/Excel uploads with templates, OAuth connectors, automated validation, guidance for missing or imperfect data
*   Question-Driven MMM Setup: simple business questions mapped to PyMC-Marketing settings, adstock and saturation modeling
*   Model Training & Transparency: progress indicators, real-time status messages, plain-English step explanations
*   Results Dashboard: overall and channel-level ROI metrics, contribution and saturation curves, interactive D3.js charts with tooltips
*   Reporting & Summaries: exportable PDF/CSV reports and AI-generated plain-language summaries
*   Budget Optimization: scenario builder, “what if” sliders, live reallocation recommendations
*   Notifications & Audit Logs: email/webhook triggers for key events, admin-viewable activity history

### 5. Tech Stack & Tools

*   Frontend: React with JavaScript (ES6+), D3.js for interactive charts, HTML5/CSS3, Material UI (or Chakra UI) for components
*   Backend: Python 3.x, Django with Django REST Framework, PyMC-Marketing for Bayesian MMM
*   Database & Storage: PostgreSQL for relational data, AWS S3 (or Replit storage) for files and model outputs
*   Auth & Email: Django Allauth for user flows, SMTP service (e.g., SendGrid) for notifications
*   Hosting & Dev Tools: Replit paid plan for compute and background tasks, Git for version control, optional IDE plugins
*   Integrations: OAuth 2.0 for Google Ads, Facebook Ads, Google Analytics; webhook support for event notifications

### 6. Non-Functional Requirements

*   Performance: dashboard loads under 2 seconds, model status updates within 5 seconds of state change
*   Scalability: modular API design to add new connectors and engines, multi-tenant schema for data isolation
*   Security & Privacy: HTTPS for all traffic (future SSL support), encryption at rest and in transit, role-based access control
*   Usability: responsive design (desktop/tablet), contextual tooltips, clear non-technical error messages
*   Compliance: GDPR-compatible data handling, audit logs for traceability

### 7. Constraints & Assumptions

*   Replit paid plan will provide sufficient CPU, memory, and background execution for MVP
*   PostgreSQL is available either in Replit or via an external cloud instance
*   PyMC-Marketing library is stable and meets modeling needs
*   SMTP or webhook endpoints will be configured by the client
*   Users will supply daily or weekly data files under 50 MB, and use modern browsers (Chrome, Firefox, Edge)

### 8. Known Issues & Potential Pitfalls

*   Large file uploads may hit browser or server limits; plan for size checks or chunked uploads
*   API rate limits from Google/Facebook may throttle data connectors; implement retries with backoff
*   Model training times can vary widely; provide clear time estimates and allow users to work elsewhere while waiting
*   Data quality variations may cause errors; mitigation through robust validation and clear user guidance
*   Replit environment limits on compute or storage; monitor usage and plan for migration to a dedicated cloud VM if needed

## App Flow Document

### Onboarding and Sign-In/Sign-Up

When a new user visits the platform, they see a clean page asking for their email and password to create an account or to log in. After they enter their information and confirm via a verification email if needed, they are signed in. A “forgot password” link sends a reset email. Signing out is always available in the header menu. Once logged in, users are brought to their organization dashboard and may switch workspaces or log out from the same menu.

### Main Dashboard or Home Page

After signing in, users land on a dashboard that lists all their organization’s workspaces as cards or rows. Each entry shows the workspace name, creation date, and last activity. A prominent “Create New Project” button lets them start a new MMM workspace. A sidebar or top navigation bar provides links to account settings, help, and, for super-admins, an Admin Console.

### Detailed Feature Flows and Page Transitions

When a user opens or creates a workspace, they are taken to a data intake page. Here, they either upload a CSV/Excel file using a drag-and-drop area or click a button to connect an API. Upload leads to a column-mapping step, and connectors trigger an OAuth flow. Once data passes validation, the user moves to a question page where they answer business-oriented prompts. Submitting those answers brings up a confirmation screen, and then model training kicks off with a progress bar. When training finishes, the app automatically navigates to the Insights Dashboard. From the dashboard, clicking “Budget Optimization” opens a scenario tool overlay. Users can switch back and forth between dashboards, reports, and optimization without losing context.

### Settings and Account Management

Users click their avatar or name in the header to access personal settings, where they can update their email, password, notification preferences, and API credentials. Organization owners can invite or remove members and view audit logs in a dedicated area. After saving changes, users click a “Back to Dashboard” link or use the main navigation to return to their projects.

### Error States and Alternate Paths

If a user uploads an invalid file or misses required columns, a clear inline error message explains the problem and suggests fixes. During API authentication failures, the app shows a retry button with a note about checking credentials. If the network connection drops, a banner alerts the user and retries in the background. Model training errors display a friendly message, log the error to the admin console, and offer a “Retry Training” button.

### Conclusion and Overall App Journey

From sign-up to insights, the user experiences a streamlined flow: authenticate, choose or create a workspace, ingest data, set up the model through simple questions, watch the training progress, and explore the results. The journey ends with actionable budget scenarios and exportable reports, all while maintaining clear paths back to settings or other workspaces.

## Tech Stack Document

### Frontend Technologies

*   React (JavaScript ES6+): Builds a responsive, component-based UI
*   D3.js: Powers interactive and animated charts for contribution and saturation curves
*   HTML5 & CSS3: Provides semantic structure and modern styling capabilities
*   Material UI (or Chakra UI): Offers ready-made, accessible UI components for forms, buttons, and layout

### Backend Technologies

*   Python 3.x: General-purpose language for model orchestration and API logic
*   Django & Django REST Framework: Enables a scalable, modular backend with built-in admin and REST endpoints
*   PyMC-Marketing: Supplies the Bayesian MMM engine with adstock and saturation modeling
*   PostgreSQL: Serves as the production-grade relational database for users, organizations, and project metadata

### Infrastructure and Deployment

*   Replit Paid Plan: Hosts the application, runs background model training, and supports collaborative development
*   AWS S3 (or Replit Storage): Stores uploaded data files and generated model artifacts securely
*   Git (via Replit): Manages source control and enables rollback if needed
*   CI/CD: Automated testing and deployment pipelines ensure code quality and fast releases

### Third-Party Integrations

*   Google Ads, Facebook Ads, Google Analytics APIs: Provide direct data connectors via OAuth 2.0 flows
*   SendGrid (SMTP) or similar: Delivers email notifications for password resets and model events
*   Webhooks: Allows external systems to receive real-time alerts on model completion or errors

### Security and Performance Considerations

*   HTTPS for all traffic (SSL planned): Secures data in transit
*   Encryption at rest for files and database backups: Protects stored data
*   Role-Based Access Control: Separates normal users from super-admins
*   Caching of read-heavy endpoints: Improves dashboard load times
*   Input validation and sanitization: Prevents injection attacks and ensures data quality

### Conclusion and Overall Tech Stack Summary

This stack leverages familiar, widely supported technologies to ensure maintainability and scalability. React and D3.js offer a snappy frontend experience, while Django and PostgreSQL provide a solid, secure backend. Replit’s hosting simplifies MVP deployment, and AWS S3 guarantees reliable file storage. Together, these choices align with our goal of delivering a polished, user-friendly MMM platform for non-technical marketers.

## Frontend Guidelines Document

### Frontend Architecture

Our frontend is a single-page React application structured around reusable components. We separate concerns by grouping files into feature folders (e.g., DataUpload, ModelTraining, Dashboard) and use a global state store for shared data. This approach supports scalability by making it easy to add new features or pages without tangled dependencies.

### Design Principles

*   Usability: Every interface element should be clear, with minimal steps to complete tasks
*   Accessibility: Follow WCAG guidelines by using semantic HTML, proper ARIA attributes, and keyboard navigation support
*   Responsiveness: Ensure layouts adapt gracefully to desktop and tablet screen sizes
*   Consistency: Use a shared theme with consistent colors, typography, and spacing

### Styling and Theming

We employ Material UI’s styling solution with a custom theme object to define primary and secondary colors, typography scales, and spacing units. This lets us switch style variants (light/dark) easily and maintain consistency. Our design leans toward a clean, flat aesthetic with occasional depth cues (shadows) to highlight interactive elements.

### Component Structure

Components live under `src/components` and are categorized by feature. Each component folder contains its JSX, styling file, and unit tests. Reusable UI pieces (buttons, inputs, cards) go in a shared `ui` folder. This structure encourages reuse and makes it straightforward for new developers or an AI assistant to find and modify components.

### State Management

We use React Context for lightweight global state (user info, theme). For complex feature states (data upload status, model training progress), we employ Redux Toolkit. This separation keeps global data lean while giving powerful tooling (middleware, devtools) for feature-specific flows.

### Routing and Navigation

React Router handles client-side routing. We define routes for `/login`, `/dashboard`, `/workspace/:id/data`, `/workspace/:id/model`, `/workspace/:id/results`, and admin pages under `/admin`. Route guards check authentication and redirect unauthorized access to the login page.

### Performance Optimization

We implement code splitting via dynamic `import()` to load feature modules only when needed. Charts and heavy visualizations are lazy-loaded. We memoize pure components and use `React.memo` or `useMemo` to prevent unnecessary re-renders. Images and icons are optimized using SVGs and WebP formats.

### Testing and Quality Assurance

*   Unit Tests: Jest and React Testing Library validate component behavior and edge cases
*   Integration Tests: Cypress simulates user flows like login, data upload, and model training
*   End-to-End Tests: We script full scenarios in Cypress to catch integration regressions
*   Linting & Formatting: ESLint and Prettier enforce code consistency

### Conclusion and Overall Frontend Summary

These guidelines ensure our frontend is modular, maintainable, and performant. By following clear design principles and structured state management, we create a user experience that is both intuitive for marketers and straightforward for developers to extend.

## Implementation Plan

1.  Set up the code repository in Replit with Django backend and React frontend boilerplate.
2.  Configure PostgreSQL and AWS S3 (or Replit storage) for data persistence and file uploads.
3.  Implement user authentication flows using Django Allauth and build the React login page.
4.  Create multi-tenant models (Organization and Workspace) and super-admin views in Django.
5.  Build the data intake page: file upload component, template download, column mapping, and validation logic.
6.  Integrate OAuth connectors for Google Ads, Facebook Ads, and Google Analytics.
7.  Develop the question-driven MMM setup UI and map answers to PyMC-Marketing configurations.
8.  Implement model training endpoint with progress updates and status broadcasting via WebSockets or polling.
9.  Design and build the interactive insights dashboard using React and D3.js.
10. Add AI-generated summary component, PDF and CSV export functionality.
11. Create the budget optimization scenario builder and wire it to model outputs.
12. Set up email notifications and webhooks for key events.
13. Develop audit logging for uploads, model runs, and user actions with an admin interface to view logs.
14. Write unit, integration, and end-to-end tests, then configure CI/CD pipelines.
15. Perform usability testing, fix bugs, and optimize performance.
16. Prepare deployment scripts, monitor resource usage, and plan post-MVP feature roadmap.

This plan provides a structured, step-by-step path to build, test, and launch the MVP while ensuring quality and scalability.
