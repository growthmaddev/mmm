import { eq, and, desc, sql } from "drizzle-orm";
import { db } from "./db";
import * as schema from "@shared/schema";

export interface IStorage {
  // User operations
  getUser(id: string): Promise<schema.User | undefined>;
  getUserByEmail(email: string): Promise<schema.User | undefined>;
  upsertUser(user: schema.UpsertUser): Promise<schema.User>;
  
  // Organization operations
  getOrganization(id: number): Promise<schema.Organization | undefined>;
  createOrganization(org: schema.InsertOrganization): Promise<schema.Organization>;
  
  // Project operations
  getProject(id: number): Promise<schema.Project | undefined>;
  getProjectsByOrganization(organizationId: number): Promise<schema.Project[]>;
  createProject(project: schema.InsertProject): Promise<schema.Project>;
  updateProject(id: number, project: Partial<schema.InsertProject>): Promise<schema.Project | undefined>;
  
  // Data Source operations
  getDataSource(id: number): Promise<schema.DataSource | undefined>;
  getDataSourcesByProject(projectId: number): Promise<schema.DataSource[]>;
  createDataSource(dataSource: schema.InsertDataSource): Promise<schema.DataSource>;
  
  // Model operations
  getModel(id: number): Promise<schema.Model | undefined>;
  getModelsByProject(projectId: number): Promise<schema.Model[]>;
  createModel(model: schema.InsertModel): Promise<schema.Model>;
  updateModel(id: number, model: Partial<schema.InsertModel>): Promise<schema.Model | undefined>;
  
  // Channel operations
  getChannel(id: number): Promise<schema.Channel | undefined>;
  getChannelsByProject(projectId: number): Promise<schema.Channel[]>;
  createChannel(channel: schema.InsertChannel): Promise<schema.Channel>;
  
  // Budget Scenario operations
  getBudgetScenario(id: number): Promise<schema.BudgetScenario | undefined>;
  getBudgetScenariosByModel(modelId: number): Promise<schema.BudgetScenario[]>;
  createBudgetScenario(scenario: schema.InsertBudgetScenario): Promise<schema.BudgetScenario>;
  
  // Audit Log operations
  createAuditLog(log: schema.InsertAuditLog): Promise<schema.AuditLog>;
  getAuditLogsByOrganization(organizationId: number, limit?: number): Promise<schema.AuditLog[]>;
}

export class DatabaseStorage implements IStorage {
  // User operations
  async getUser(id: string): Promise<schema.User | undefined> {
    const [user] = await db.select().from(schema.users).where(eq(schema.users.id, id));
    return user;
  }
  
  async getUserByEmail(email: string): Promise<schema.User | undefined> {
    if (!email) return undefined;
    const [user] = await db.select().from(schema.users).where(eq(schema.users.email, email));
    return user;
  }

  async upsertUser(userData: schema.UpsertUser): Promise<schema.User> {
    const [user] = await db
      .insert(schema.users)
      .values(userData)
      .onConflictDoUpdate({
        target: schema.users.id,
        set: {
          ...userData,
          updatedAt: new Date(),
        },
      })
      .returning();
    return user;
  }
  
  // Organization operations
  async getOrganization(id: number): Promise<schema.Organization | undefined> {
    const [org] = await db.select().from(schema.organizations).where(eq(schema.organizations.id, id));
    return org;
  }
  
  async createOrganization(org: schema.InsertOrganization): Promise<schema.Organization> {
    const [newOrg] = await db
      .insert(schema.organizations)
      .values(org)
      .returning();
    return newOrg;
  }
  
  // Project operations
  async getProject(id: number): Promise<schema.Project | undefined> {
    const [project] = await db.select().from(schema.projects).where(eq(schema.projects.id, id));
    return project;
  }
  
  async getProjectsByOrganization(organizationId: number): Promise<schema.Project[]> {
    return await db
      .select()
      .from(schema.projects)
      .where(eq(schema.projects.organizationId, organizationId))
      .orderBy(desc(schema.projects.updatedAt));
  }
  
  async createProject(project: schema.InsertProject): Promise<schema.Project> {
    const [newProject] = await db
      .insert(schema.projects)
      .values(project)
      .returning();
    return newProject;
  }
  
  async updateProject(id: number, project: Partial<schema.InsertProject>): Promise<schema.Project | undefined> {
    const [updatedProject] = await db
      .update(schema.projects)
      .set({...project, updatedAt: new Date()})
      .where(eq(schema.projects.id, id))
      .returning();
    return updatedProject;
  }
  
  // Data Source operations
  async getDataSource(id: number): Promise<schema.DataSource | undefined> {
    const [dataSource] = await db.select().from(schema.dataSources).where(eq(schema.dataSources.id, id));
    return dataSource;
  }
  
  async getDataSourcesByProject(projectId: number): Promise<schema.DataSource[]> {
    return await db
      .select()
      .from(schema.dataSources)
      .where(eq(schema.dataSources.projectId, projectId));
  }
  
  async createDataSource(dataSource: schema.InsertDataSource): Promise<schema.DataSource> {
    const [newDataSource] = await db
      .insert(schema.dataSources)
      .values(dataSource)
      .returning();
    return newDataSource;
  }
  
  // Model operations
  async getModel(id: number): Promise<schema.Model | undefined> {
    const [model] = await db.select().from(schema.models).where(eq(schema.models.id, id));
    return model;
  }
  
  async getModelsByProject(projectId: number): Promise<schema.Model[]> {
    return await db
      .select()
      .from(schema.models)
      .where(eq(schema.models.projectId, projectId))
      .orderBy(desc(schema.models.updatedAt));
  }
  
  async createModel(model: schema.InsertModel): Promise<schema.Model> {
    const [newModel] = await db
      .insert(schema.models)
      .values(model)
      .returning();
    return newModel;
  }
  
  async updateModel(id: number, model: Partial<schema.InsertModel>): Promise<schema.Model | undefined> {
    const [updatedModel] = await db
      .update(schema.models)
      .set({...model, updatedAt: new Date()})
      .where(eq(schema.models.id, id))
      .returning();
    return updatedModel;
  }
  
  // Channel operations
  async getChannel(id: number): Promise<schema.Channel | undefined> {
    const [channel] = await db.select().from(schema.channels).where(eq(schema.channels.id, id));
    return channel;
  }
  
  async getChannelsByProject(projectId: number): Promise<schema.Channel[]> {
    return await db
      .select()
      .from(schema.channels)
      .where(eq(schema.channels.projectId, projectId));
  }
  
  async createChannel(channel: schema.InsertChannel): Promise<schema.Channel> {
    const [newChannel] = await db
      .insert(schema.channels)
      .values(channel)
      .returning();
    return newChannel;
  }
  
  // Budget Scenario operations
  async getBudgetScenario(id: number): Promise<schema.BudgetScenario | undefined> {
    const [scenario] = await db.select().from(schema.budgetScenarios).where(eq(schema.budgetScenarios.id, id));
    return scenario;
  }
  
  async getBudgetScenariosByModel(modelId: number): Promise<schema.BudgetScenario[]> {
    return await db
      .select()
      .from(schema.budgetScenarios)
      .where(eq(schema.budgetScenarios.modelId, modelId))
      .orderBy(desc(schema.budgetScenarios.updatedAt));
  }
  
  async createBudgetScenario(scenario: schema.InsertBudgetScenario): Promise<schema.BudgetScenario> {
    const [newScenario] = await db
      .insert(schema.budgetScenarios)
      .values(scenario)
      .returning();
    return newScenario;
  }
  
  // Audit Log operations
  async createAuditLog(log: schema.InsertAuditLog): Promise<schema.AuditLog> {
    const [newLog] = await db
      .insert(schema.auditLogs)
      .values(log)
      .returning();
    return newLog;
  }
  
  async getAuditLogsByOrganization(organizationId: number, limit: number = 100): Promise<schema.AuditLog[]> {
    return await db
      .select()
      .from(schema.auditLogs)
      .where(eq(schema.auditLogs.organizationId, organizationId))
      .orderBy(desc(schema.auditLogs.timestamp))
      .limit(limit);
  }
}

export const storage = new DatabaseStorage();
