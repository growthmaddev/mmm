const bcrypt = require('bcryptjs');
const { db } = require('./server/db');
const { users } = require('./shared/schema');

async function createTestUser() {
  try {
    // Create a test user with email and password
    const hashedPassword = await bcrypt.hash('test123', 10);
    
    await db.insert(users).values({
      id: 'test-user-id',
      email: 'test@example.com',
      firstName: 'Test',
      lastName: 'User',
      createdAt: new Date(),
      updatedAt: new Date()
    }).onConflictDoUpdate({
      target: users.id,
      set: {
        email: 'test@example.com',
        updatedAt: new Date()
      }
    });
    
    console.log('Test user created successfully!');
    console.log('Email: test@example.com');
    console.log('Password: test123');
  } catch (error) {
    console.error('Error creating test user:', error);
  } finally {
    process.exit(0);
  }
}

createTestUser();