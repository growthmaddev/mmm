import {
  pgTable,
  text,
  serial,
  integer,
  boolean,
  timestamp,
  varchar,
  json,
  jsonb,
  unique,
  foreignKey,
  pgEnum,
  index,
} from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Enums
export const projectStatusEnum = pgEnum('project_status', [
  'draft',
  'uploading_data',
  'configuring_model',
  'training',
  'completed',
  'error'
]);

export const modelTrainingStatusEnum = pgEnum('model_training_status', [
  'queued',
  'preprocessing',
  'training',
  'postprocessing',
  'completed',
  'error'
]);

export const channelTypeEnum = pgEnum('channel_type', [
  'display',
  'search',
  'social',
  'email',
  'tv',
  'radio',
  'print',
  'ooh',
  'other'
]);

export const dataSourceEnum = pgEnum('data_source', [
  'csv_upload',
  'google_ads',
  'facebook_ads',
  'google_analytics',
  'manual_entry'
]);

// Organizations
export const organizations = pgTable("organizations", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Users
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  email: text("email").notNull().unique(),
  password: text("password").notNull(),
  firstName: text("first_name"),
  lastName: text("last_name"),
  organizationId: integer("organization_id").references(() => organizations.id),
  role: text("role").default("user").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Projects
export const projects = pgTable("projects", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  description: text("description"),
  organizationId: integer("organization_id").notNull().references(() => organizations.id),
  createdById: integer("created_by_id").notNull().references(() => users.id),
  status: projectStatusEnum("status").default("draft").notNull(),
  dateRange: json("date_range").$type<{startDate: string, endDate: string}>(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Data Sources
export const dataSources = pgTable("data_sources", {
  id: serial("id").primaryKey(),
  projectId: integer("project_id").notNull().references(() => projects.id),
  type: dataSourceEnum("type").notNull(),
  fileName: text("file_name"),
  fileUrl: text("file_url"),
  connectionInfo: json("connection_info"),
  dateColumn: text("date_column"),
  metricColumns: json("metric_columns").$type<string[]>(),
  channelColumns: json("channel_columns").$type<{[key: string]: string}>(),
  controlColumns: json("control_columns").$type<{[key: string]: string}>(),
  createdById: integer("created_by_id").notNull().references(() => users.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// MMM Models (configurations and results)
export const models = pgTable("models", {
  id: serial("id").primaryKey(),
  projectId: integer("project_id").notNull().references(() => projects.id),
  name: text("name").notNull(),
  status: modelTrainingStatusEnum("status").default("queued").notNull(),
  progress: integer("progress").default(0),
  adstockSettings: json("adstock_settings"),
  saturationSettings: json("saturation_settings"),
  controlVariables: json("control_variables"),
  responseVariables: json("response_variables"),
  results: json("results"),
  createdById: integer("created_by_id").notNull().references(() => users.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Marketing Channels
export const channels = pgTable("channels", {
  id: serial("id").primaryKey(),
  projectId: integer("project_id").notNull().references(() => projects.id),
  name: text("name").notNull(),
  type: channelTypeEnum("type").notNull(),
  spend: json("spend").$type<{[date: string]: number}>(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
}, (table) => {
  return {
    unq: unique().on(table.projectId, table.name),
  };
});

// Budget Optimization Scenarios
export const budgetScenarios = pgTable("budget_scenarios", {
  id: serial("id").primaryKey(),
  modelId: integer("model_id").notNull().references(() => models.id),
  name: text("name").notNull(),
  totalBudget: integer("total_budget").notNull(),
  allocations: json("allocations").$type<{[channelId: string]: number}>(),
  projectedResults: json("projected_results"),
  createdById: integer("created_by_id").notNull().references(() => users.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Audit Logs
export const auditLogs = pgTable("audit_logs", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id),
  organizationId: integer("organization_id").references(() => organizations.id),
  action: text("action").notNull(),
  details: json("details"),
  ipAddress: text("ip_address"),
  userAgent: text("user_agent"),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
});

// Schema insertion types
export const insertUserSchema = createInsertSchema(users).omit({
  id: true,
  createdAt: true,
  updatedAt: true
});

export const insertOrganizationSchema = createInsertSchema(organizations).omit({
  id: true,
  createdAt: true,
  updatedAt: true
});

export const insertProjectSchema = createInsertSchema(projects).omit({
  id: true,
  createdAt: true,
  updatedAt: true
});

export const insertDataSourceSchema = createInsertSchema(dataSources).omit({
  id: true,
  createdAt: true,
  updatedAt: true
});

export const insertModelSchema = createInsertSchema(models).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
  status: true,
  progress: true,
  results: true
});

export const insertChannelSchema = createInsertSchema(channels).omit({
  id: true,
  createdAt: true,
  updatedAt: true
});

export const insertBudgetScenarioSchema = createInsertSchema(budgetScenarios).omit({
  id: true,
  createdAt: true,
  updatedAt: true
});

export const insertAuditLogSchema = createInsertSchema(auditLogs).omit({
  id: true,
  timestamp: true
});

// Types
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

export type InsertOrganization = z.infer<typeof insertOrganizationSchema>;
export type Organization = typeof organizations.$inferSelect;

export type InsertProject = z.infer<typeof insertProjectSchema>;
export type Project = typeof projects.$inferSelect;

export type InsertDataSource = z.infer<typeof insertDataSourceSchema>;
export type DataSource = typeof dataSources.$inferSelect;

export type InsertModel = z.infer<typeof insertModelSchema>;
export type Model = typeof models.$inferSelect;

export type InsertChannel = z.infer<typeof insertChannelSchema>;
export type Channel = typeof channels.$inferSelect;

export type InsertBudgetScenario = z.infer<typeof insertBudgetScenarioSchema>;
export type BudgetScenario = typeof budgetScenarios.$inferSelect;

export type InsertAuditLog = z.infer<typeof insertAuditLogSchema>;
export type AuditLog = typeof auditLogs.$inferSelect;
