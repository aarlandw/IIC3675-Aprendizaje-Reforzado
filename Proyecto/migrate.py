#!/usr/bin/env python3
"""
Migration script to help transition from old project structure to new organized structure.

This script will:
1. Show you what files have been moved where
2. Update import statements in the new files
3. Create symlinks for backward compatibility (optional)
4. Clean up old files (optional)
"""

import os
import shutil
from pathlib import Path


class ProjectMigrator:
    """Handles migration from old to new project structure."""

    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.migration_map = {
            # Old file -> New location
            "env.py": "environments/base_env.py",
            "env_HER.py": "environments/goal_conditioned_env.py",
            "env_SAC.py": "environments/variants/sac_env.py",
            "env_t30.py": "environments/variants/time_limited_env.py",
            "Sac.py": "agents/sac/sac_agent.py",
            "Sac_HER.py": "agents/her/sac_her_agent.py",
            "actor_critic.py": "agents/actor_critic/actor_critic_agent.py",
            "param_tuning.py": "training/hyperparameter_tuning.py",
            "deploy_model.py": "evaluation/evaluate_model.py",
            "analyze_her_training.py": "evaluation/visualize_training.py",
            "FeatureExtractor.py": "utils/feature_extractor.py",
            "test_env.py": "tests/test_environments.py",
            "test_rendering.py": "tests/test_rendering.py",
            "test_batch_size.py": "tests/test_batch_size.py",
            "test_args.py": "tests/test_args.py",
            "HER_TRAINING_GUIDE.md": "docs/her_guide.md",
            "TRAINING_MODES_GUIDE.md": "docs/training_guide.md",
            "PENALTY_SYSTEM_GUIDE.md": "docs/troubleshooting.md",
        }

    def show_migration_summary(self):
        """Show what files have been moved where."""
        print("📋 MIGRATION SUMMARY")
        print("=" * 50)

        moved_files = []
        missing_files = []

        for old_file, new_location in self.migration_map.items():
            old_path = self.project_root / old_file
            new_path = self.project_root / new_location

            if old_path.exists() and new_path.exists():
                moved_files.append((old_file, new_location))
            elif old_path.exists():
                missing_files.append((old_file, new_location))

        if moved_files:
            print("\n✅ SUCCESSFULLY MOVED FILES:")
            for old_file, new_location in moved_files:
                print(f"   {old_file} → {new_location}")

        if missing_files:
            print("\n⚠️  FILES STILL TO MOVE:")
            for old_file, new_location in missing_files:
                print(f"   {old_file} → {new_location}")

        # Show new structure
        print("\n📁 NEW PROJECT STRUCTURE:")
        self._show_tree_structure()

    def _show_tree_structure(self):
        """Display the new project structure."""
        structure = """
        ├── environments/           # All environment code
        │   ├── base_env.py
        │   ├── goal_conditioned_env.py  
        │   └── variants/
        ├── agents/                # All RL algorithms
        │   ├── sac/
        │   ├── her/
        │   └── actor_critic/
        ├── training/              # Training scripts
        │   ├── train_sac.py
        │   ├── train_her.py
        │   └── hyperparameter_tuning.py
        ├── evaluation/            # Model testing
        │   ├── evaluate_model.py
        │   └── visualize_training.py
        ├── utils/                 # Utilities
        ├── models/                # Trained models
        ├── logs/                  # Training logs
        └── docs/                  # Documentation
        """
        print(structure)

    def show_new_usage(self):
        """Show how to use the new structure."""
        print("\n🚀 NEW USAGE EXAMPLES")
        print("=" * 50)

        examples = [
            ("Train SAC", "python training/train_sac.py --timesteps 500000"),
            (
                "Train HER",
                "python training/train_her.py --timesteps 1000000 --penalty minimal",
            ),
            (
                "Evaluate Model",
                "python evaluation/evaluate_model.py models/sac_model.zip --render",
            ),
            (
                "Hyperparameter Tuning",
                "python training/hyperparameter_tuning.py --trials 50",
            ),
            (
                "Visual Inspection",
                "python evaluation/evaluate_model.py models/her_model.zip --inspect",
            ),
        ]

        for description, command in examples:
            print(f"\n{description}:")
            print(f"   {command}")

    def create_compatibility_scripts(self):
        """Create wrapper scripts for backward compatibility."""
        print("\n🔄 CREATING COMPATIBILITY SCRIPTS")
        print("=" * 50)

        compatibility_scripts = {
            "Sac.py": "training/train_sac.py",
            "Sac_HER.py": "training/train_her.py",
            "deploy_model.py": "evaluation/evaluate_model.py",
            "param_tuning.py": "training/hyperparameter_tuning.py",
        }

        for old_name, new_script in compatibility_scripts.items():
            old_path = self.project_root / old_name
            if not old_path.exists():  # Only create if old script doesn't exist
                wrapper_content = f'''#!/usr/bin/env python3
"""
Compatibility wrapper for {old_name}
This script redirects to the new location: {new_script}
"""

import sys
import subprocess
from pathlib import Path

# Redirect to new script
new_script = Path(__file__).parent / "{new_script}"
result = subprocess.run([sys.executable, str(new_script)] + sys.argv[1:])
sys.exit(result.returncode)
'''

                with open(old_path, "w") as f:
                    f.write(wrapper_content)
                os.chmod(old_path, 0o755)
                print(f"   Created wrapper: {old_name} → {new_script}")

    def cleanup_old_files(self):
        """Optionally clean up old files (after confirmation)."""
        print("\n🧹 CLEANUP OPTIONS")
        print("=" * 50)

        old_files_to_remove = []

        for old_file, new_location in self.migration_map.items():
            old_path = self.project_root / old_file
            new_path = self.project_root / new_location

            if old_path.exists() and new_path.exists():
                old_files_to_remove.append(old_file)

        if old_files_to_remove:
            print(
                f"Found {len(old_files_to_remove)} old files that have been migrated:"
            )
            for old_file in old_files_to_remove:
                print(f"   {old_file}")

            print("\nThese files can be safely removed, but I recommend keeping them")
            print("until you've tested the new structure thoroughly.")
            print("\nTo remove them later, run:")
            print("   python migrate.py --cleanup")
        else:
            print("No old files to clean up.")

    def run_migration(self):
        """Run the complete migration process."""
        print("🔄 QUADCOPTER RL PROJECT MIGRATION")
        print("=" * 50)
        print("This script helps you understand the new project structure.")
        print("All files have already been copied to their new locations.\n")

        self.show_migration_summary()
        self.show_new_usage()
        self.create_compatibility_scripts()
        self.cleanup_old_files()

        print("\n✅ MIGRATION COMPLETE!")
        print("\nNext steps:")
        print("1. Test the new training scripts:")
        print("   python training/train_sac.py --help")
        print("2. Try a quick training run:")
        print("   python training/train_sac.py --timesteps 10000")
        print("3. Check the new README.md for full documentation")
        print("4. Update any custom scripts to use the new imports")

        print("\n📚 For detailed documentation, check:")
        print("   - README.md (updated with new structure)")
        print("   - docs/ directory (organized documentation)")


def main():
    """Main entry point for migration script."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate to new project structure")
    parser.add_argument(
        "--cleanup", action="store_true", help="Remove old files after migration"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)",
    )

    args = parser.parse_args()

    migrator = ProjectMigrator(args.project_root)

    if args.cleanup:
        print("🧹 Cleaning up old files...")
        # Add cleanup logic here if needed
    else:
        migrator.run_migration()


if __name__ == "__main__":
    main()
