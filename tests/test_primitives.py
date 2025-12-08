"""Tests for evals.primitives module."""

from evals.primitives import ConversationHistory, Turn


class TestTurn:
    """Tests for the Turn dataclass."""

    def test_turn_creation(self):
        """Test basic turn creation."""
        turn = Turn(role="user", content="Hello, world!")
        assert turn.role == "user"
        assert turn.content == "Hello, world!"
        assert turn.template_vars == {}
        assert turn.metadata == {}

    def test_turn_with_metadata(self):
        """Test turn creation with metadata."""
        turn = Turn(
            role="assistant",
            content="Hi there!",
            metadata={"turn_index": 0, "cached": True},
        )
        assert turn.metadata["turn_index"] == 0
        assert turn.metadata["cached"] is True

    def test_turn_render_simple(self):
        """Test rendering a turn with variables."""
        turn = Turn(role="user", content="Hello, {{ name }}!")
        rendered = turn.render({"name": "Alice"})

        assert rendered.content == "Hello, Alice!"
        assert rendered.template_vars == {"name": "Alice"}
        assert rendered.role == "user"

    def test_turn_render_multiple_vars(self):
        """Test rendering with multiple variables."""
        turn = Turn(
            role="user",
            content="Set assertiveness to {{ level }} and tone to {{ tone }}.",
        )
        rendered = turn.render({"level": 0.8, "tone": "formal"})

        assert "0.8" in rendered.content
        assert "formal" in rendered.content

    def test_turn_render_preserves_metadata(self):
        """Test that rendering preserves metadata."""
        turn = Turn(
            role="user",
            content="{{ greeting }}",
            metadata={"important": True},
        )
        rendered = turn.render({"greeting": "Hello"})

        assert rendered.metadata["important"] is True

    def test_turn_to_message_dict(self):
        """Test conversion to OpenAI message format."""
        turn = Turn(role="assistant", content="I can help with that.")
        msg = turn.to_message_dict()

        assert msg == {"role": "assistant", "content": "I can help with that."}

    def test_turn_repr(self):
        """Test string representation."""
        turn = Turn(role="user", content="Short message")
        repr_str = repr(turn)

        assert "user" in repr_str
        assert "Short message" in repr_str

    def test_turn_repr_truncates_long_content(self):
        """Test that repr truncates long content."""
        long_content = "A" * 100
        turn = Turn(role="user", content=long_content)
        repr_str = repr(turn)

        assert len(repr_str) < 100
        assert "..." in repr_str


class TestConversationHistory:
    """Tests for the ConversationHistory class."""

    def test_empty_history(self):
        """Test creating empty history."""
        history = ConversationHistory()
        assert len(history) == 0
        assert history.turns == []

    def test_add_turn(self):
        """Test adding turns."""
        history = ConversationHistory()
        turn = Turn(role="user", content="Hello")
        history.add_turn(turn)

        assert len(history) == 1
        assert history[0].content == "Hello"

    def test_add_user(self):
        """Test add_user convenience method."""
        history = ConversationHistory()
        history.add_user("What is 2+2?")

        assert len(history) == 1
        assert history[0].role == "user"
        assert history[0].content == "What is 2+2?"

    def test_add_assistant(self):
        """Test add_assistant convenience method."""
        history = ConversationHistory()
        history.add_assistant("The answer is 4.")

        assert len(history) == 1
        assert history[0].role == "assistant"

    def test_add_system(self):
        """Test add_system convenience method."""
        history = ConversationHistory()
        history.add_system("You are a helpful assistant.")

        assert len(history) == 1
        assert history[0].role == "system"

    def test_to_messages(self):
        """Test conversion to OpenAI messages format."""
        history = ConversationHistory()
        history.add_system("You are helpful.")
        history.add_user("Hello")
        history.add_assistant("Hi there!")

        messages = history.to_messages()

        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Hello"}
        assert messages[2] == {"role": "assistant", "content": "Hi there!"}

    def test_render_all(self):
        """Test rendering all turns with variables."""
        history = ConversationHistory()
        history.add_user("My name is {{ name }}")
        history.add_assistant("Hello, {{ name }}!")

        rendered = history.render_all({"name": "Bob"})

        assert rendered[0].content == "My name is Bob"
        assert rendered[1].content == "Hello, Bob!"

    def test_reversed(self):
        """Test reversing user turns for hysteresis probes."""
        history = ConversationHistory()
        history.add_system("System prompt")
        history.add_user("First question")
        history.add_user("Second question")
        history.add_user("Third question")

        reversed_history = history.reversed()

        # System should stay first
        assert reversed_history[0].role == "system"
        # User turns should be reversed
        assert reversed_history[1].content == "Third question"
        assert reversed_history[2].content == "Second question"
        assert reversed_history[3].content == "First question"

    def test_reversed_metadata(self):
        """Test that reversed history has metadata flag."""
        history = ConversationHistory()
        history.add_user("Question")

        reversed_history = history.reversed()

        assert reversed_history.metadata.get("reversed") is True

    def test_iteration(self):
        """Test iterating over history."""
        history = ConversationHistory()
        history.add_user("One")
        history.add_user("Two")

        contents = [turn.content for turn in history]

        assert contents == ["One", "Two"]

    def test_indexing(self):
        """Test indexing into history."""
        history = ConversationHistory()
        history.add_user("First")
        history.add_user("Second")

        assert history[0].content == "First"
        assert history[1].content == "Second"
