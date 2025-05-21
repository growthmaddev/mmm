import { eq, and, desc, sql } from "drizzle-orm";
import { db } from "./db";
import * as schema from "@shared/schema";

export interface IStorage {
  // User operations
  getUser(id: number): Promise<schema.User | undefined>;
  getUserByEmail(email: string): Promise<schema.User | undefined>;
  createUser(user: schema.RegisterUser): Promise<schema.User>;
  
  // Organization operations
  getOrganization(id: number): Promise<schema.Organization | undefined>;
  createOrganization(org: schema.InsertOrganization): Promise<schema.Organization>;
  
  // Project operations
  getProject(id: number): Promise<schema.Project | undefined>;
  getProjectsByOrganization(organizationId: number): Promise<schema.Project[]>;
  createProject(project: schema.InsertProject): Promise<schema.Project>;
  updateProject(id: number, project: Partial<schema.InsertProject>): Promise<schema.Project | undefined>;
  deleteProject(id: number): Promise<boolean>;
  
  // Data Source operations
  getDataSource(id: number): Promise<schema.DataSource | undefined>;
  getDataSourcesByProject(projectId: number): Promise<schema.DataSource[]>;
  createDataSource(dataSource: schema.InsertDataSource): Promise<schema.DataSource>;
  updateDataSource(id: number, dataSource: Partial<schema.UpdateDataSource>): Promise<schema.DataSource | undefined>;
  
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
  async getUser(id: number): Promise<schema.User | undefined> {
    const [user] = await db.select().from(schema.users).where(eq(schema.users.id, id));
    return user;
  }
  
  async getUserByEmail(email: string): Promise<schema.User | undefined> {
    if (!email) return undefined;
    const [user] = await db.select().from(schema.users).where(eq(schema.users.email, email));
    return user;
  }

  async createUser(userData: schema.RegisterUser): Promise<schema.User> {
    const [user] = await db
      .insert(schema.users)
      .values({
        email: userData.email,
        password: userData.password,
        username: userData.username,
        firstName: userData.firstName,
        lastName: userData.lastName,
        role: 'user'
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
  
  async deleteProject(id: number): Promise<boolean> {
    try {
      // First, delete related records (to maintain referential integrity)
      // Delete models associated with the project
      await db
        .delete(schema.models)
        .where(eq(schema.models.projectId, id));
      
      // Delete data sources
      await db
        .delete(schema.dataSources)
        .where(eq(schema.dataSources.projectId, id));
        
      // Delete channels
      await db
        .delete(schema.channels)
        .where(eq(schema.channels.projectId, id));
      
      // Finally, delete the project itself
      await db
        .delete(schema.projects)
        .where(eq(schema.projects.id, id));
      
      return true;
    } catch (error) {
      console.error("Error deleting project:", error);
      return false;
    }
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
      .values({
        projectId: dataSource.projectId,
        type: dataSource.type,
        fileName: dataSource.fileName,
        fileUrl: dataSource.fileUrl,
        connectionInfo: dataSource.connectionInfo || {},
        dateColumn: dataSource.dateColumn,
        metricColumns: dataSource.metricColumns,
        channelColumns: dataSource.channelColumns,
        controlColumns: dataSource.controlColumns,
        createdById: dataSource.createdById,
      })
      .returning();
    return newDataSource;
  }
  
  async updateDataSource(id: number, dataSource: Partial<schema.UpdateDataSource>): Promise<schema.DataSource | undefined> {
    const updateData: Record<string, any> = {
      updatedAt: new Date()
    };
    
    // Handle each property individually to avoid type errors
    if (dataSource.fileName !== undefined) updateData.fileName = dataSource.fileName;
    if (dataSource.fileUrl !== undefined) updateData.fileUrl = dataSource.fileUrl;
    if (dataSource.connectionInfo !== undefined) updateData.connectionInfo = dataSource.connectionInfo;
    if (dataSource.dateColumn !== undefined) updateData.dateColumn = dataSource.dateColumn;
    if (dataSource.metricColumns !== undefined) updateData.metricColumns = dataSource.metricColumns;
    if (dataSource.channelColumns !== undefined) updateData.channelColumns = dataSource.channelColumns;
    if (dataSource.controlColumns !== undefined) updateData.controlColumns = dataSource.controlColumns;
    
    const [updatedDataSource] = await db
      .update(schema.dataSources)
      .set(updateData)
      .where(eq(schema.dataSources.id, id))
      .returning();
    
    return updatedDataSource;
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
  
  async updateModel(id: number, model: schema.UpdateModel): Promise<schema.Model | undefined> {
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
